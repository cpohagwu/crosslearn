from __future__ import annotations

import torch


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve ``auto`` to the best available torch device."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_device_map(device_map: str | torch.device) -> str:
    """Resolve Chronos/HF-style device strings to a concrete target."""
    return str(resolve_device(device_map))
