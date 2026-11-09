# Audio2Text Local

Local audio transcription service using state-of-the-art deep learning models with CUDA GPU acceleration.

## Features

- GPU-accelerated audio transcription using CUDA
- Support for multiple audio formats (MP3, WAV, etc.)
- Containerized deployment with Docker
- Automatic audio format conversion using FFmpeg
- Efficient processing with PyTorch and Transformers

## Prerequisites

### System Requirements

- NVIDIA GPU with CUDA support (Compute Capability 3.5 or higher recommended)
- Linux system (Ubuntu 22.04 or similar)
- At least 8GB GPU memory (for most models)

### Software Requirements

1. **NVIDIA Drivers**
   ```bash
   # Check if NVIDIA drivers are installed
   nvidia-smi
   ```
   If not installed, follow [NVIDIA Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)

2. **Docker** (version 20.10 or higher)
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Add your user to docker group
   sudo usermod -aG docker $USER
   newgrp docker
   ```

3. **NVIDIA Container Toolkit**
   ```bash
   # Add NVIDIA package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   # Install nvidia-docker2
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2

   # Restart Docker daemon
   sudo systemctl restart docker

   # Verify installation
   docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi
   ```

## Installation

### 1. Clone or Setup Project

```bash
cd /path/to/audio2text-local
```

### 2. Configure Environment

Create a `.env` file for any configuration:

```bash
# Example .env file
MODEL_NAME=openai/whisper-large-v3
DEVICE=cuda
```

### 3a. Run Locally with uv

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv with Python 3.13 and install locked dependencies
uv sync

# Run
uv run python main.py

# Optional: enable git hooks
uv run pre-commit install
```

On Linux, `torch` is installed from the PyTorch CUDA 13.2 index; on macOS the regular
PyPI build (CPU / Apple MPS) is used. FFmpeg must be installed on the system for non-WAV input.

Add or upgrade dependencies with `uv add <package>` / `uv lock --upgrade`.

### 3b. Build Docker Image

```bash
docker build -t audio2text-cuda .
```

This will:
- Pull the minimal NVIDIA CUDA 13.2 base image (Ubuntu 24.04)
- Install Python 3.13 and the locked dependencies with uv
- Install PyTorch with bundled CUDA libraries
- Set up FFmpeg for audio processing

## Usage

### Running with Docker and CUDA

#### Basic Run

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  audio2text-cuda
```

#### Run with Specific GPU

```bash
# Use GPU 0
docker run --gpus '"device=0"' \
  -v $(pwd)/data:/app/data \
  audio2text-cuda

# Use GPUs 0 and 1
docker run --gpus '"device=0,1"' \
  -v $(pwd)/data:/app/data \
  audio2text-cuda
```

#### Run with Environment Variables

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -e MODEL_NAME=openai/whisper-large-v3 \
  -e DEVICE=cuda \
  audio2text-cuda
```

#### Interactive Mode

```bash
docker run --gpus all -it \
  -v $(pwd)/data:/app/data \
  audio2text-cuda bash
```

#### Run in Background (Detached)

```bash
docker run --gpus all -d \
  --name audio2text \
  -v $(pwd)/data:/app/data \
  audio2text-cuda

# View logs
docker logs -f audio2text

# Stop container
docker stop audio2text

# Remove container
docker rm audio2text
```

### Volume Mounting

Mount your audio files directory:

```bash
docker run --gpus all \
  -v /path/to/your/audio:/app/data \
  audio2text-cuda
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
services:
  audio2text:
    build: .
    image: audio2text-cuda
    container_name: audio2text
    volumes:
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Run with:
```bash
docker compose up
```

## Verification

### Check GPU Availability

After starting the container, verify GPU is accessible:

```bash
docker run --gpus all -it audio2text-cuda python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA Available: True
CUDA Device: NVIDIA GeForce RTX 3090
```

## Troubleshooting

### GPU Not Detected

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```

2. **Verify Docker can access GPU:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi
   ```

3. **Check NVIDIA Container Toolkit:**
   ```bash
   dpkg -l | grep nvidia-docker
   ```

### Out of Memory Errors

- Reduce batch size in your processing script
- Use a smaller model variant
- Close other GPU-intensive applications
- Monitor GPU memory:
  ```bash
  watch -n 1 nvidia-smi
  ```

### Permission Denied Errors

```bash
# Ensure user is in docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo
sudo docker run --gpus all audio2text-cuda
```

### Container Exits Immediately

Check logs:
```bash
docker logs $(docker ps -lq)
```

### CUDA Version Mismatch

The image uses PyTorch built for CUDA 13.2, which needs a recent NVIDIA driver (580+).
Ensure your driver supports CUDA 13.x:
```bash
nvidia-smi | grep "CUDA Version"
```

If your driver only supports an older CUDA version, switch the PyTorch index in
`pyproject.toml` (e.g. `https://download.pytorch.org/whl/cu128`), run `uv lock`,
and change the Dockerfile base image to a matching tag (e.g. `nvidia/cuda:12.8.1-base-ubuntu24.04`).

## Performance Optimization

### GPU Memory Management

Add to your Python code:
```python
import torch

# Clear cache
torch.cuda.empty_cache()

# Set memory growth
torch.cuda.set_per_process_memory_fraction(0.8)
```

### Multi-GPU Setup

For multi-GPU processing, use:
```bash
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  audio2text-cuda
```

## Dependencies

- Python 3.13 (managed by uv)
- PyTorch 2.15 (CUDA 13.2 build on Linux)
- Transformers 5.22
- librosa 1.0, pydub (+ audioop-lts)
- FFmpeg
- See `pyproject.toml` / `uv.lock` for the full list

## Project Structure

```
audio2text-local/
├── Dockerfile          # CUDA-enabled Docker configuration
├── pyproject.toml      # Project metadata and dependencies (uv)
├── uv.lock             # Locked dependency versions
├── .python-version     # Python version used by uv
├── main.py            # Main application
├── src/
│   ├── utils.py       # Utility functions
│   └── logger_config.py  # Logging configuration
├── data/              # Audio files directory
└── .env               # Environment variables (optional)
```

## License

MIT License

## Author

Zmeykoo (skrypaslav@gmail.com)
