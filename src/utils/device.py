"""Elegir el mejor dispositivo de cómputo disponible.

Prioridad: MPS (GPU de Apple Silicon) > CUDA (GPU NVIDIA, p.ej. en Colab) > CPU.
"""
import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
