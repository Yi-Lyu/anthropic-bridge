import argparse
import os

import uvicorn

from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic Bridge Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    copilot_token = os.environ.get("GITHUB_COPILOT_TOKEN", "")

    app = create_app(
        openrouter_api_key=api_key or None,
        copilot_token=copilot_token or None,
    )

    print(f"Starting Anthropic Bridge on {args.host}:{args.port}")
    static_bearer = os.environ.get("OPENAI_RESPONSES_API_KEY", "")
    endpoint = os.environ.get(
        "OPENAI_RESPONSES_ENDPOINT",
        "https://chatgpt.com/backend-api/codex/responses",
    )
    override_model = os.environ.get("OPENAI_RESPONSES_MODEL_OVERRIDE", "")
    if static_bearer:
        print(f"  OpenAI: static-bearer mode → {endpoint}")
        if override_model:
            print(f"          model override: * → {override_model}")
    else:
        print("  OpenAI: OAuth mode (reads ~/.codex/auth.json)")
    if copilot_token:
        print("  Copilot: copilot/* models")
    else:
        print("  Copilot: disabled (set GITHUB_COPILOT_TOKEN)")
    if api_key:
        print("  OpenRouter: openrouter/* models")
    else:
        print("  OpenRouter: disabled (set OPENROUTER_API_KEY)")
    # Graceful shutdown: give in-flight streaming requests (up to ~300s of
    # codex inference) a chance to drain before uvicorn kills them. systemd's
    # TimeoutStopSec (15s on our unit) will still force-kill if something is
    # stuck beyond this, so 60s is the ceiling the process will wait.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_graceful_shutdown=60,
    )


if __name__ == "__main__":
    main()
