# Multi-stage build for anthropic-bridge
# - Stage 1: install deps + app into a venv using uv
# - Stage 2: slim runtime image

FROM python:3.12-slim AS builder

# Install uv (fast Python installer)
RUN pip install --no-cache-dir uv==0.5.18

WORKDIR /build

# Copy project metadata first for better layer caching
COPY pyproject.toml uv.lock* ./

# Sync deps into a venv at /venv (no dev/test extras in production)
RUN uv venv /venv && \
    VIRTUAL_ENV=/venv uv pip install --no-cache -e .

# Copy the rest of the source
COPY anthropic_bridge/ ./anthropic_bridge/

# Reinstall to pick up the source (editable install already points to the right dir)
RUN VIRTUAL_ENV=/venv uv pip install --no-cache .


FROM python:3.12-slim

# Runtime deps only (httpx / aiofiles / fastapi / uvicorn / tiktoken + bridge code are in /venv)
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder /venv /venv
COPY --from=builder /build/anthropic_bridge /app/anthropic_bridge

# Health endpoint lives at /health (bridge returns {"status": "ok"})
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3); sys.exit(0)"

EXPOSE 8080

ENTRYPOINT ["python", "-m", "anthropic_bridge"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
