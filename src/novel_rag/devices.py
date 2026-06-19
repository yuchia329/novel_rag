"""Resolve a requested torch device, falling back to CPU when unavailable."""

from __future__ import annotations


def resolve_device(device: str) -> str:
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device
