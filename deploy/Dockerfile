FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv
FROM python:3.13.15-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26 AS builder-base
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./

FROM python:3.13.15-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26 AS runtime-base
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/tmp/huggingface TOKENIZERS_PARALLELISM=false \
    ENCODER_MODEL_ROOT=/models
RUN groupadd --gid 10001 encoder && useradd --uid 10001 --gid encoder --no-create-home encoder
WORKDIR /app
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).read()"]
CMD ["python", "-m", "uranus_research_encoder", "--host", "0.0.0.0", "--port", "8080"]

FROM builder-base AS builder-onnx
RUN uv sync --locked --no-default-groups --group runtime-onnx --no-install-project
COPY src ./src
RUN uv sync --locked --no-default-groups --group runtime-onnx --no-editable

FROM runtime-base AS runtime-onnx
ENV ENCODER_BACKEND=onnx
COPY --from=builder-onnx --chown=10001:10001 /app/.venv /app/.venv

FROM builder-base AS builder-torch
RUN uv sync --locked --no-default-groups --group runtime-torch --no-install-project
COPY src ./src
RUN uv sync --locked --no-default-groups --group runtime-torch --no-editable

FROM runtime-base AS runtime-torch
ENV ENCODER_BACKEND=torch
COPY --from=builder-torch --chown=10001:10001 /app/.venv /app/.venv

# Keep the default image on the reference until parity AND performance justify migration.
FROM runtime-torch AS runtime
