import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .access_log import log_event
from .cache import get_reasoning_cache
from .protocol import collect_anthropic_response, estimate_anthropic_input_tokens
from .providers import CopilotProvider, OpenAIProvider, OpenRouterProvider
from .providers.openai.auth import auth_file_exists


async def _cancel_on_client_disconnect(
    request: Request,
    upstream: AsyncIterator[str],
    req_id: str | None,
) -> AsyncIterator[str]:
    """Wrap an SSE generator so that if the client goes away, the upstream
    httpx request inside ``upstream`` is promptly closed instead of continuing
    to burn tokens. We do this by polling ``request.is_disconnected()`` in a
    background task and calling ``upstream.aclose()`` in finally, which
    propagates cancellation to the provider's ``async with httpx.AsyncClient``.

    Note: ``is_disconnected()`` is less reliable under BaseHTTPMiddleware,
    which is what FastAPI's ``@app.middleware("http")`` uses. The finally-
    path ``aclose()`` is the real guarantee — even without watcher firing,
    when FastAPI decides the request is over, it closes the async generator,
    which runs our finally block.
    """
    disconnected = asyncio.Event()

    async def _watch() -> None:
        try:
            while not disconnected.is_set():
                if await request.is_disconnected():
                    disconnected.set()
                    return
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    watcher = asyncio.create_task(_watch())
    emitted_bytes = 0
    try:
        async for chunk in upstream:
            if disconnected.is_set():
                log_event({
                    "level": "warn",
                    "action": "client_disconnected_mid_stream",
                    "req_id": req_id,
                    "emitted_bytes": emitted_bytes,
                })
                break
            emitted_bytes += len(chunk)
            yield chunk
    finally:
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await upstream.aclose()
        except Exception:
            pass

ProviderType = Literal["openrouter", "copilot", "openai"]


@dataclass
class ProxyConfig:
    openrouter_api_key: str | None = None
    copilot_token: str | None = None


class AnthropicBridge:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.app = FastAPI(title="Anthropic Bridge")
        self._openrouter_clients: dict[str, OpenRouterProvider] = {}
        self._openai_clients: dict[str, OpenAIProvider] = {}
        self._copilot_clients: dict[str, CopilotProvider] = {}
        self._setup_routes()
        self._setup_cors()
        get_reasoning_cache()

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Structured access log (one JSON event per request completion).
        # Emits to the file at $ANTHROPIC_BRIDGE_LOG_FILE and stdout.
        # Business fields (model, has_tools, usage, streaming) are emitted
        # from the provider so this middleware stays cheap and non-intrusive.
        @self.app.middleware("http")
        async def _access_log(request: Request, call_next):
            req_id = uuid.uuid4().hex[:12]
            request.state.req_id = req_id
            start = time.perf_counter()
            client_ip = request.client.host if request.client else None

            status = 500
            error_msg: str | None = None
            try:
                response = await call_next(request)
                status = response.status_code
                return response
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                raise
            finally:
                duration_ms = int((time.perf_counter() - start) * 1000)
                event: dict = {
                    "level": "info" if (status < 500 and not error_msg) else "error",
                    "action": "http_access",
                    "req_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "duration_ms": duration_ms,
                    "client_ip": client_ip,
                }
                if error_msg:
                    event["error"] = error_msg
                log_event(event)

    def _setup_routes(self) -> None:
        @self.app.get("/")
        async def root() -> dict[str, str]:
            return {"status": "ok", "message": "Anthropic Bridge"}

        @self.app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self.app.post("/v1/messages/count_tokens")
        async def count_tokens(request: Request) -> JSONResponse:
            body = await request.json()
            return JSONResponse({"input_tokens": estimate_anthropic_input_tokens(body)})

        @self.app.post("/v1/messages", response_model=None)
        async def messages(request: Request) -> StreamingResponse | JSONResponse:
            body = await request.json()
            model = body.get("model", "")
            req_id = getattr(request.state, "req_id", None)

            # Business-level access event. Pairs with the middleware's
            # "http_access" event via req_id. Captures what the Anthropic
            # client requested (model, stream, tools, message count) —
            # distinct from the upstream-side model which the provider
            # may override via OPENAI_RESPONSES_MODEL_OVERRIDE.
            messages_list = body.get("messages") or []
            tools_list = body.get("tools") or []
            # Defensive truncation on client-controlled strings to keep
            # log lines bounded regardless of what callers send.
            safe_model = str(model)[:128] if model else ""
            log_event(
                {
                    "level": "info",
                    "action": "bridge_request",
                    "req_id": req_id,
                    "anthropic_model": safe_model,
                    "streaming": body.get("stream") is True,
                    "has_tools": len(tools_list) > 0,
                    "messages_count": len(messages_list),
                    "system_present": body.get("system") is not None,
                }
            )

            provider = self._get_provider(model)
            if provider is None:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "type": "authentication_error",
                            "message": self._get_provider_error_message(model),
                        }
                    },
                )

            if body.get("stream") is not True:
                message, error = await collect_anthropic_response(provider.handle(body))
                if error:
                    return JSONResponse(status_code=502, content=error)
                if message is None:
                    return JSONResponse(
                        status_code=502,
                        content={
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "Provider returned no message.",
                            },
                        },
                    )
                return JSONResponse(message)

            return StreamingResponse(
                _cancel_on_client_disconnect(request, provider.handle(body), req_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

    def _get_provider(
        self, model: str
    ) -> CopilotProvider | OpenAIProvider | OpenRouterProvider | None:
        requested_provider = self._get_requested_provider(model)

        if requested_provider:
            requested_model = self._model_for_provider(model, requested_provider)
            return self._make_provider(requested_model, requested_provider)

        for provider_type in ("openrouter", "copilot", "openai"):
            provider_model = self._model_for_provider(model, provider_type)
            provider = self._make_provider(provider_model, provider_type)
            if provider:
                return provider

        return None

    def _get_provider_error_message(self, model: str) -> str:
        requested_provider = self._get_requested_provider(model)
        if requested_provider is None:
            return (
                "No provider configured. Set OPENROUTER_API_KEY, "
                "GITHUB_COPILOT_TOKEN, or configure OpenAI auth."
            )

        messages = {
            "openai": "OpenAI auth not configured. Run 'codex login' to use openai/* models.",
            "copilot": "GitHub Copilot token not configured. Set GITHUB_COPILOT_TOKEN to use copilot/* models.",
            "openrouter": "OpenRouter API key not configured. Set OPENROUTER_API_KEY to use openrouter/* models.",
        }
        return messages[requested_provider]

    def _get_requested_provider(self, model: str) -> ProviderType | None:
        if model.startswith("openai/"):
            return "openai"
        if model.startswith("copilot/"):
            return "copilot"
        if model.startswith("openrouter/"):
            return "openrouter"
        return None

    def _model_for_provider(self, model: str, provider_type: ProviderType) -> str:
        model_name = model.split("/", 1)[1] if "/" in model else model
        if provider_type == "openrouter":
            return f"openrouter/{model_name}"
        if provider_type == "copilot":
            return f"copilot/{model_name}"
        return f"openai/{model_name}"

    def _make_provider(
        self, model: str, provider_type: ProviderType
    ) -> CopilotProvider | OpenAIProvider | OpenRouterProvider | None:
        if provider_type == "openrouter" and self.config.openrouter_api_key:
            if model not in self._openrouter_clients:
                self._openrouter_clients[model] = OpenRouterProvider(
                    model, self.config.openrouter_api_key
                )
            return self._openrouter_clients[model]

        if provider_type == "copilot" and self.config.copilot_token:
            if model not in self._copilot_clients:
                self._copilot_clients[model] = CopilotProvider(
                    model, self.config.copilot_token
                )
            return self._copilot_clients[model]

        if provider_type == "openai" and auth_file_exists():
            if model not in self._openai_clients:
                self._openai_clients[model] = OpenAIProvider(model)
            return self._openai_clients[model]

        return None


def create_app(
    openrouter_api_key: str | None = None,
    copilot_token: str | None = None,
) -> FastAPI:
    config = ProxyConfig(
        openrouter_api_key=openrouter_api_key,
        copilot_token=copilot_token,
    )
    return AnthropicBridge(config).app
