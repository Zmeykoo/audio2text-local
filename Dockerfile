# Every PyTorch build used here bundles its own GPU runtime (CUDA / ROCm / XPU),
# so a plain Ubuntu base is enough - the host only needs the GPU driver.
FROM ubuntu:24.04

# uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /uvx /bin/

# ffmpeg is required for pydub audio conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA_* vars are read by the NVIDIA Container Toolkit and ignored elsewhere
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# PyTorch build: cuda (NVIDIA, default; also runs on CPU) | rocm (AMD) | xpu (Intel) | cpu
ARG TORCH_BACKEND=cuda
# AMD only: GPU architecture to bundle kernels for, e.g. gfx1100 (RX 7900), gfx1201 (RX 9070)
ARG ROCM_ARCH=""

# Install dependencies first (cached layer); uv downloads Python 3.13 itself
COPY pyproject.toml uv.lock .python-version ./
COPY scripts/ ./scripts/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-default-groups --group ${TORCH_BACKEND} \
    && if [ "${TORCH_BACKEND}" = "rocm" ]; then \
        if [ -z "${ROCM_ARCH}" ]; then echo "Set --build-arg ROCM_ARCH=gfxXXXX for the rocm image" >&2; exit 1; fi; \
        python scripts/setup.py --kernels-only --arch "${ROCM_ARCH}"; \
    fi

COPY main.py ./
COPY src/ ./src/

RUN mkdir -p /app/data

# Usage: docker run ... <image> data/<file>
ENTRYPOINT ["python", "main.py"]
