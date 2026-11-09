# Torch wheels from the cu132 index bundle the CUDA runtime libraries,
# so the minimal "base" CUDA image is enough (the host only needs the driver).
FROM nvidia/cuda:13.2.0-base-ubuntu24.04

# uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /uvx /bin/

# ffmpeg is required for pydub audio conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Install dependencies first (cached layer), uv downloads Python 3.13 itself
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY main.py ./
COPY src/ ./src/

RUN mkdir -p /app/data

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "main.py"]
