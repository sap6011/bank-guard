# syntax=docker/dockerfile:1

# The BankGuard scoring service: FastAPI, the sequence transformer on CPU-only
# torch, and the demo bundle. Build from the repo root:
#   docker build -t bankguard-api .

ARG PYTHON_IMAGE=python:3.13-slim

# Build stage: install the service into a virtual environment with uv.
FROM ${PYTHON_IMAGE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# 1. Main dependencies only, exactly as locked: no dev tools, no CUDA torch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-install-project

# 2. CPU-only torch, the version the model was trained and evaluated with.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

# 3. Our package, installed as a regular package; --inexact keeps torch.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable --inexact

# Runtime stage: a clean slim Python with the environment and the model files.
FROM ${PYTHON_IMAGE}
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app models/sequence/preprocessing.json models/sequence/sequence_seed*.pt /app/models/sequence/
COPY --chown=app:app models/demo/bundle.npz /app/models/demo/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    BANKGUARD_MODEL_DIR=/app/models/sequence \
    BANKGUARD_BUNDLE=/app/models/demo/bundle.npz \
    PORT=8000
USER app
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn bankguard.api:app --host 0.0.0.0 --port ${PORT}"]