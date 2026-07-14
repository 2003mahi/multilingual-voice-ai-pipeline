"""GPU detection, memory tracking, and device resolution."""
from __future__ import annotations

import torch


def resolve_device(preferred: str = "auto") -> str:
    """Return 'cuda' / 'cuda:N' / 'cpu'."""
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    # MPS (Apple) is not well supported by these TTS models → fall back to CPU
    return "cpu"


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def gpu_memory_mb() -> float:
    """Peak allocated GPU memory in MB (0.0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_utilization() -> float:
    """Best-effort GPU utilization % via pynvml; 0.0 if unavailable."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        return 0.0
