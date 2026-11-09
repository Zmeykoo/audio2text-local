# Audio2Text Local

Local audio transcription with OpenAI Whisper (via Hugging Face Transformers).
Runs on Linux, Windows and macOS. It uses whatever accelerator it finds and falls back to the CPU automatically.

## Features

- Automatic device selection: NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Apple Silicon (MPS), NPU plugins, or CPU
- Automatic recovery from out-of-memory errors: smaller batch first, then CPU
- Any audio/video format FFmpeg can read (MP3, M4A, WAV, MP4, ...)
- Output as JSON with timestamps
- One setup command that installs the right PyTorch build for your GPU on Linux, Windows and macOS
- Reproducible installs with [uv](https://docs.astral.sh/uv/), and Docker images for NVIDIA, AMD and CPU

## How device selection works

You don't need to configure a device. On startup the app:

1. Detects the available accelerator through `torch.accelerator` (CUDA/ROCm, XPU, MPS, NPU plugins). If none is found, it uses the CPU.
2. Loads the model in `float16` on the accelerator (`float32` on CPU).
3. Picks a starting batch size from the free accelerator memory.
4. If the accelerator **runs out of memory**, it frees the cache and retries with half the batch size, down to 1.
5. If that still fails, or the backend raises any other error (driver issue, unsupported op, ...), it **moves the model to the CPU** and finishes there.

On Apple Silicon, ops that MPS doesn't implement automatically run on the CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`).

> Whisper always processes 30-second windows, so a smaller chunk length doesn't reduce memory.
> Batch size is the setting that matters, and the app adjusts it automatically.

## Configuration

All settings are optional. Set them as environment variables or in a `.env` file in the project root:

```bash
MODEL_NAME=openai/whisper-large-v3   # any Whisper model on Hugging Face, e.g. openai/whisper-small
AUDIO_PATH=data/meeting.m4a          # used when no file is passed on the command line
DEVICE=auto                          # override only if you really want: auto | cuda | xpu | mps | npu | cpu
DEBUG_MODE=false                     # true = debug logs + logs.log file
```

`whisper-large-v3` needs ~4 GB of GPU memory in fp16 (~7 GB of RAM on CPU). On weaker machines, use
`openai/whisper-small` or `openai/whisper-base`.

Transcripts are written to `data/<file-name>_<id>.json`.

---

## Quick start

1. Install [uv](https://docs.astral.sh/uv/) and FFmpeg (see your OS below).
2. Install the GPU driver if you have a GPU (see your OS below).
3. From the project folder, let the setup script pick the right PyTorch build for your hardware:

   ```bash
   uv run --no-project scripts/setup.py
   ```

   | Detected hardware | Installed build |
   |---|---|
   | NVIDIA GPU | CUDA 13.2 |
   | AMD GPU | ROCm 10.1 + GPU kernels for your exact card (e.g. `gfx1100`) |
   | Intel Arc / Core Ultra GPU | XPU |
   | No GPU | small CPU-only build |
   | macOS | default build (Apple GPU via MPS) |

   To force a choice, pass `--backend cuda|rocm|xpu|cpu`. For AMD, `--arch gfx1100` skips GPU detection.

4. Transcribe. The setup script prints the exact command for your build:

   ```bash
   uv run python main.py path/to/audio.mp3            # NVIDIA, macOS
   uv run --no-sync python main.py path/to/audio.mp3  # AMD, Intel, CPU-only
   ```

   `--no-sync` keeps the build you installed. Without it, `uv run` switches back to the default
   NVIDIA build. You can also set `UV_NO_SYNC=1` once in your shell, or activate `.venv` and run `python main.py`.

## macOS (Apple Silicon or Intel)

Apple Silicon Macs use the GPU through Metal (MPS) automatically. Intel Macs run on the CPU.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install ffmpeg

uv run --no-project scripts/setup.py     # same as `uv sync` on macOS
uv run python main.py path/to/audio.m4a
```

Docker on macOS can't access the Apple GPU, so run natively for GPU speed.

## Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt install ffmpeg          # Fedora: sudo dnf install ffmpeg, Arch: sudo pacman -S ffmpeg

uv run --no-project scripts/setup.py
```

**NVIDIA:** install a recent driver (580+, supports CUDA 13.x) and check it with `nvidia-smi`.
You don't need the CUDA toolkit, because PyTorch bundles it.

**AMD:** you only need the `amdgpu` kernel driver, which ships with modern kernels (Ubuntu 24.04+, Fedora 40+, ...).
You don't need a system ROCm install, because the ROCm runtime comes from pip. Give your user GPU access, then log out and back in:

```bash
sudo usermod -aG render,video $USER
```

The setup script reads the GPU architecture from the driver (`/sys/class/kfd`) or AMD's `offload-arch` tool.
Supported: Radeon RX 6000 (gfx103x), RX 7000 (gfx110x), RX 9000 (gfx120x), Ryzen AI APUs (gfx115x),
Instinct MI100/MI200/MI300/MI350 (gfx908/gfx90a/gfx942/gfx950).

**Intel GPU:** install the [Intel GPU drivers](https://dgpu-docs.intel.com/driver/installation.html).

## Windows

Run these in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
winget install --id Gyan.FFmpeg -e
# restart the terminal so uv and ffmpeg are on PATH

uv run --no-project scripts/setup.py
uv run python main.py C:\path\to\audio.m4a   # add --no-sync for AMD / Intel / CPU builds
```

**NVIDIA:** install the latest [Game Ready / Studio driver](https://www.nvidia.com/drivers) (580+), then check it with `nvidia-smi`.

**AMD:** install the latest [AMD Software: Adrenalin Edition](https://www.amd.com/en/support/download/drivers.html) driver.
Native Windows ROCm supports Radeon RX 6000 / 7000 / 9000 and Ryzen AI APUs (gfx103x, gfx110x, gfx115x, gfx120x).
The setup script finds the architecture with AMD's `offload-arch` tool or through PyTorch.
If it can't, look up your GPU's `gfx` code and pass it, e.g. `--arch gfx1100`.

**Intel GPU:** install the latest Intel Arc & Iris Xe graphics driver.

---

## Docker

Pick the image for your GPU. Every image falls back to the CPU when no GPU is available.

```bash
docker build -t audio2text-cuda .                                         # NVIDIA (default)
docker build --build-arg TORCH_BACKEND=rocm --build-arg ROCM_ARCH=gfx1100 -t audio2text-rocm .   # AMD
docker build --build-arg TORCH_BACKEND=cpu -t audio2text-cpu .            # CPU only, smaller
```

Because Docker can't see the GPU while building, the AMD image needs your GPU's architecture as `ROCM_ARCH`. On Linux, find it with:
`cat /sys/class/kfd/kfd/topology/nodes/*/properties | grep gfx_target_version` (`110000` means `gfx1100`).

Put audio files in `./data`, then:

```bash
# NVIDIA: Linux, or Windows with Docker Desktop (WSL2)
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" -v hf-cache:/app/.cache/huggingface \
  audio2text-cuda data/audio.m4a

# AMD: Linux only
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v "$(pwd)/data:/app/data" -v hf-cache:/app/.cache/huggingface \
  audio2text-rocm data/audio.m4a

# CPU: any OS, including macOS
docker run --rm \
  -v "$(pwd)/data:/app/data" -v hf-cache:/app/.cache/huggingface \
  audio2text-cpu data/audio.m4a
```

The `hf-cache` volume keeps downloaded models between runs. Pass settings with `-e`, e.g. `-e MODEL_NAME=openai/whisper-small`.
In Windows PowerShell, replace `$(pwd)` with `${PWD}`.

**GPU in Docker requirements:**
- NVIDIA on Linux: driver plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- NVIDIA on Windows: Docker Desktop with the WSL2 backend and a current NVIDIA driver. No extra toolkit is needed.
- AMD: Linux host with the `amdgpu` driver. On Windows, run natively instead, because Docker can't pass AMD GPUs through.
- macOS: no GPU passthrough. Use the CPU image or run natively.

Docker Compose example (NVIDIA):

```yaml
services:
  audio2text:
    build: .
    image: audio2text-cuda
    command: ["data/audio.m4a"]
    volumes:
      - ./data:/app/data
      - hf-cache:/app/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
volumes:
  hf-cache:
```

For AMD, replace `deploy:` with `devices: ["/dev/kfd", "/dev/dri"]` and `group_add: [video, render]`, and build with the ROCm args.

---

## Verify which device is used

```bash
uv run --no-sync python -c "import torch; print(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu')"
```

The app also logs the selected device on startup, e.g. `Detected device: CUDA: NVIDIA GeForce RTX 3090` or
`Detected device: ROCm: AMD Radeon RX 7900 XTX (gfx1100)`.

## Troubleshooting

- **Runs on CPU although I have a GPU:** check the driver and rerun `uv run --no-project scripts/setup.py`.
  If you ran plain `uv run` after installing an AMD/Intel/CPU build, uv switched back to the NVIDIA build. Rerun setup and use `uv run --no-sync`.
- **AMD: "GPU kernels for your card are probably missing":** run `uv run --no-project scripts/setup.py --backend rocm`.
  Also rerun it after any `uv sync`, because `uv sync` removes the per-GPU kernel packages.
- **AMD: architecture not detected or not supported:** pass it with `--arch gfxXXXX`. Cards older than RX 6000 (Vega, Polaris) have no ROCm kernels and run on the CPU.
- **Very slow on CPU:** use a smaller model, e.g. `MODEL_NAME=openai/whisper-small`.
- **`ffmpeg` not found:** install FFmpeg (see your OS section above) and restart the terminal.
- **Old NVIDIA driver (below 580):** switch the `pytorch-cu132` index URL in `pyproject.toml` to
  `https://download.pytorch.org/whl/cu128` and run `uv lock`.

## Development

```bash
uv run --no-project scripts/setup.py   # includes dev tools (pre-commit, black); --no-dev to skip
uv run pre-commit install
uv add <package>             # add a dependency
uv lock --upgrade            # upgrade all locked versions
```

## Project Structure

```
audio2text-local/
├── Dockerfile          # Docker image (CUDA, ROCm, XPU or CPU build of PyTorch)
├── pyproject.toml      # Project metadata, dependencies, PyTorch build groups (uv)
├── uv.lock             # Locked dependency versions for all platforms
├── .python-version     # Python version used by uv
├── main.py             # Entry point: transcription + fallback logic
├── scripts/
│   └── setup.py        # Detects hardware and installs the matching PyTorch build
├── src/
│   ├── device.py         # Accelerator detection and memory helpers
│   ├── utils.py          # Audio conversion, transcript writing
│   └── logger_config.py  # Logging configuration
├── data/               # Audio files and transcripts
└── .env                # Environment variables (optional)
```

## Dependencies

- Python 3.13 (managed by uv)
- PyTorch 2.15: CUDA 13.2 / XPU / CPU builds on Linux and Windows, PyPI build on macOS
- PyTorch 2.14 + ROCm 10.1 for AMD GPUs (AMD's own build, Linux and Windows)
- Transformers 5.22, Accelerate 1.15
- librosa 1.0, pydub (+ audioop-lts)
- FFmpeg (system package)

## License

MIT License

## Author

Zmeykoo (skrypaslav@gmail.com)
