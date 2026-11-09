"""Install the right PyTorch build for this machine.

Usage (from the project root, works on Linux, Windows and macOS):

    uv run --no-project scripts/setup.py              # auto-detect hardware
    uv run --no-project scripts/setup.py --backend cpu
    uv run --no-project scripts/setup.py --backend rocm --arch gfx1100

What it does:
  * macOS                -> default build (Apple GPU via MPS, or CPU)
  * NVIDIA GPU           -> CUDA build
  * AMD GPU              -> ROCm build + GPU kernels for your exact GPU (gfxXXXX)
  * Intel Arc / Core Ultra GPU -> XPU build
  * nothing found        -> small CPU-only build

Only uses the standard library, so it can run before dependencies are installed.
"""

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ("auto", "cuda", "rocm", "xpu", "cpu")

TORCH_PROBE = """
import json, importlib.metadata as md, torch
info = {"torch": torch.__version__, "hip": torch.version.hip,
        "extras": md.metadata("torch").get_all("Provides-Extra") or []}
try:
    info["arch"] = torch.cuda.get_device_properties(0).gcnArchName
except Exception:
    info["arch"] = None
print("PROBE:" + json.dumps(info))
"""


def log(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check)


def capture(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def uv_exe() -> str:
    exe = os.environ.get("UV") or shutil.which("uv")
    if not exe:
        sys.exit("uv not found: install it from https://docs.astral.sh/uv/")
    return exe


def amd_index_url() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for index in pyproject["tool"]["uv"]["index"]:
        if index["name"] == "amd-rocm":
            return index["url"]
    sys.exit("amd-rocm index missing from pyproject.toml")


def venv_python() -> str:
    venv = Path(os.environ.get("UV_PROJECT_ENVIRONMENT", ROOT / ".venv"))
    exe = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(exe) if exe.exists() else sys.executable


# --------------------------------------------------------------------------- #
# Hardware detection (no torch needed)
# --------------------------------------------------------------------------- #
def windows_gpu_names() -> list[str]:
    out = capture(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
        ]
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def linux_gpu_names() -> list[str]:
    out = capture(["lspci", "-mm"])
    return [
        line
        for line in out.splitlines()
        if "VGA" in line or "Display" in line or "3D" in line
    ]


def linux_kfd_archs() -> list[str]:
    """AMD GPUs known to the amdgpu kernel driver, as gfx names."""
    archs = []
    for props in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties"):
        try:
            text = Path(props).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("gfx_target_version"):
                version = int(line.split()[1])
                if version:
                    archs.append(gfx_from_target_version(version))
    return archs


def gfx_from_target_version(version: int) -> str:
    """110000 -> gfx1100, 90010 -> gfx90a, 120001 -> gfx1201."""
    major, minor, step = version // 10000, (version // 100) % 100, version % 100
    return f"gfx{major}{minor:x}{step:x}"


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and "GPU" in capture(
        ["nvidia-smi", "-L"]
    )


def detect_backend() -> str:
    system = platform.system()
    if system == "Darwin":
        return "default"  # PyPI torch supports Apple GPUs (MPS) and CPU
    if has_nvidia_gpu():
        log("NVIDIA GPU detected")
        return "cuda"

    names = windows_gpu_names() if system == "Windows" else linux_gpu_names()
    lowered = " | ".join(names).lower()
    if (
        (system == "Linux" and linux_kfd_archs())
        or "radeon" in lowered
        or "amd" in lowered
    ):
        log("AMD GPU detected")
        return "rocm"
    if "intel" in lowered and ("arc" in lowered or "data center gpu" in lowered):
        log("Intel GPU detected")
        return "xpu"
    log("No supported GPU detected, using CPU build")
    return "cpu"


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #
def sync(backend: str, dev: bool) -> None:
    cmd = [uv_exe(), "sync"]
    if backend not in ("cuda", "default"):  # cuda is the default group
        cmd += ["--no-default-groups", "--group", backend]
        if dev:
            cmd += ["--group", "dev"]
    elif not dev:
        cmd += ["--no-dev"]
    run(cmd)


def probe_torch(python: str) -> dict:
    out = subprocess.run(
        [python, "-c", TORCH_PROBE], capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        if line.startswith("PROBE:"):
            return json.loads(line[len("PROBE:") :])
    sys.exit("Could not import torch from the project environment")


def offload_arch_archs(python: str) -> list[str]:
    """AMD's own detection tool, installed into the venv by rocm-sdk-core."""
    scripts = Path(python).parent
    for exe in (
        scripts / "offload-arch",
        scripts / "offload-arch.exe",
        shutil.which("offload-arch"),
    ):
        if exe and Path(exe).exists():
            return capture([str(exe)]).split()
    return []


def detect_amd_archs(python: str, probe: dict) -> list[str]:
    archs = offload_arch_archs(python)
    if platform.system() == "Linux":
        archs += linux_kfd_archs()
    if probe.get("arch"):
        archs.append(probe["arch"])
    # "gfx90a:sramecc+:xnack-" -> "gfx90a"; keep order, drop duplicates and placeholders
    seen = []
    for arch in (a.split(":")[0].strip().lower() for a in archs):
        if arch.startswith("gfx") and arch != "gfx000" and arch not in seen:
            seen.append(arch)
    return seen


def install_rocm_kernels(python: str, arch: str | None) -> bool:
    """Add GPU kernels for this AMD GPU. Returns False if the GPU isn't supported."""
    probe = probe_torch(python)
    if not probe.get("hip"):
        sys.exit(
            "The installed torch is not a ROCm build; run without --kernels-only first"
        )

    supported = sorted(
        e.removeprefix("device-") for e in probe["extras"] if e.startswith("device-gfx")
    )
    candidates = [arch.lower()] if arch else detect_amd_archs(python, probe)
    if not candidates:
        log("Could not detect the AMD GPU architecture.")
        log("Find it in AMD's GPU list and rerun with e.g. --arch gfx1100")
        log("Supported: " + " ".join(supported))
        sys.exit(1)

    match = next((a for a in candidates if a in supported), None)
    if match is None:
        log(
            f"Detected {', '.join(candidates)}, which ROCm {probe['torch']} has no kernels for."
        )
        log("Supported: " + " ".join(supported))
        return False

    log(f"Installing ROCm GPU kernels for {match}")
    run(
        [
            uv_exe(),
            "pip",
            "install",
            "--no-config",  # use AMD's real package metadata, not the project overrides
            "--python",
            python,
            "--index-url",
            amd_index_url(),
            f"torch[device-{match}]=={probe['torch']}",
        ]
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help="PyTorch build (default: auto-detect)",
    )
    parser.add_argument(
        "--arch", help="AMD GPU architecture, e.g. gfx1100 (default: auto-detect)"
    )
    parser.add_argument(
        "--no-dev", action="store_true", help="skip dev tools (pre-commit, black)"
    )
    parser.add_argument(
        "--kernels-only",
        action="store_true",
        help="only add AMD GPU kernels to the current environment (used by the Dockerfile)",
    )
    args = parser.parse_args()

    if args.kernels_only:
        if not install_rocm_kernels(sys.executable, args.arch):
            sys.exit(1)
        return

    backend = detect_backend() if args.backend == "auto" else args.backend
    if platform.system() == "Darwin":
        if backend not in ("default", "cpu"):
            log(
                f"--backend {backend} is not available on macOS, using the default build"
            )
        backend = "default"
    if backend == "rocm" and platform.machine().lower() not in ("x86_64", "amd64"):
        log("ROCm builds exist only for x86_64, using the CPU build")
        backend = "cpu"

    sync(backend, dev=not args.no_dev)

    if backend == "rocm" and not install_rocm_kernels(venv_python(), args.arch):
        log("Falling back to the CPU build")
        backend = "cpu"
        sync(backend, dev=not args.no_dev)

    no_sync = "" if backend in ("cuda", "default") else "--no-sync "
    log(f"Done ({backend} build). Run with:")
    log(f"    uv run {no_sync}python main.py path/to/audio.mp3")
    if no_sync:
        log(
            "`--no-sync` keeps this build; plain `uv run` would switch back to the default CUDA build."
        )


if __name__ == "__main__":
    main()
