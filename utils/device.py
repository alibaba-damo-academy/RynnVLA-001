"""Device selection shared by the inference entry points (XPU, then CUDA, then CPU).

An explicit device always wins over detection and is never swapped for another backend.
"""

from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in _ACCELERATORS:
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve ``None`` to the detected default, anything else verbatim.

    A bare ``xpu`` or ``cuda`` becomes index 0.
    """
    if device is None or device == "":
        device_type = default_device_type()
        return torch.device("cpu") if device_type == "cpu" else torch.device(device_type, 0)
    device = torch.device(device)
    if device.type in _ACCELERATORS and device.index is None:
        return torch.device(device.type, 0)
    return device


def device_type(device: Union[DeviceLike, torch.Tensor, torch.nn.Module]) -> str:
    """Backend name (``cpu``, ``cuda`` or ``xpu``) of a device, tensor or module."""
    if isinstance(device, torch.nn.Module):
        device = next(device.parameters()).device
    elif isinstance(device, torch.Tensor):
        device = device.device
    return torch.device(device).type


def autocast_enabled(device: DeviceLike, dtype: torch.dtype) -> bool:
    """CUDA keeps autocast on as before, XPU rejects float32 autocast, CPU stays off."""
    dev_type = device_type(device)
    return dev_type == "cuda" or (dev_type == "xpu" and dtype != torch.float32)


def empty_cache(device: Optional[DeviceLike] = None) -> None:
    """Release cached allocator blocks of the backend of ``device`` (no-op on CPU)."""
    dev_type = device_type(device) if device is not None else default_device_type()
    if dev_type in _ACCELERATORS and is_available(dev_type):
        getattr(torch, dev_type).empty_cache()
