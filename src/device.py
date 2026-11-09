"""Accelerator detection and memory helpers.

Works with every backend PyTorch exposes through ``torch.accelerator``:
NVIDIA CUDA, AMD ROCm (reported as ``cuda``), Intel XPU, Apple MPS and
out-of-tree NPU plugins (e.g. ``torch_npu``). Falls back to CPU.
"""

import gc
import logging
import os

import torch

VALID_DEVICES = ("auto", "cpu", "cuda", "xpu", "mps", "npu")


def detect_device(requested: str | None = None) -> torch.device:
    """Pick the device to run on.

    ``requested`` comes from the optional ``DEVICE`` env var. ``auto`` (default)
    selects the accelerator PyTorch detects, or CPU when there is none.
    """
    requested = (requested or "auto").strip().lower()
    if requested not in VALID_DEVICES:
        logging.warning(f"Unknown DEVICE={requested!r}, using auto-detection")
        requested = "auto"

    if requested == "cpu":
        return torch.device("cpu")

    if torch.accelerator.is_available():
        accelerator = torch.accelerator.current_accelerator()
        if requested in ("auto", accelerator.type):
            return accelerator
        logging.warning(
            f"DEVICE={requested} requested, but detected accelerator is "
            f"{accelerator.type}; using {accelerator.type}"
        )
        return accelerator

    if requested != "auto":
        logging.warning(f"DEVICE={requested} requested, but it is not available")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cpu":
        return f"CPU ({os.cpu_count()} logical cores)"
    name = device.type
    try:
        if device.type == "cuda" and torch.version.hip:
            props = torch.cuda.get_device_properties(device)
            name = f"ROCm: {props.name} ({props.gcnArchName.split(':')[0]})"
        elif device.type == "cuda":
            name = f"CUDA: {torch.cuda.get_device_name(device)}"
        elif device.type == "xpu":
            name = f"XPU: {torch.xpu.get_device_name(device)}"
        elif device.type == "mps":
            name = "Apple Silicon GPU (MPS)"
    except Exception:  # device name is cosmetic only
        pass
    return name


def free_memory_gb(device: torch.device) -> float | None:
    """Free memory on the accelerator in GB, or None if unknown."""
    if device.type == "cpu":
        return None
    try:
        free, _total = torch.accelerator.get_memory_info(device)
        return free / 1024**3
    except Exception:
        return None


def release_memory(device: torch.device) -> None:
    gc.collect()
    if device.type != "cpu":
        try:
            torch.accelerator.empty_cache()
        except Exception:
            pass


def is_out_of_memory(error: BaseException) -> bool:
    """CUDA/XPU raise OutOfMemoryError; MPS and plugins raise a RuntimeError."""
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def accelerator_hint(error: BaseException) -> str | None:
    """Actionable advice for accelerator failures that a reinstall can fix."""
    if is_out_of_memory(error):
        return None
    if torch.version.hip:
        return (
            "AMD GPU failed - the GPU kernels for your card are probably missing. "
            "Run `uv run --no-project scripts/setup.py --backend rocm` to install them."
        )
    message = str(error).lower()
    if torch.version.cuda and ("no kernel image" in message or "driver" in message):
        return (
            "NVIDIA GPU failed - your driver is probably too old for this CUDA build. "
            "Update the NVIDIA driver (580+) and try again."
        )
    return None
