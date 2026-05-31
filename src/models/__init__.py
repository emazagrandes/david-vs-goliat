"""Fábrica de modelos: traduce un nombre de condición a un modelo.

De momento: "small" (David) y "large" (Goliat). Cuando exista el módulo
cortical, aquí añadiremos "small+cortical", etc. (condiciones B y D).
"""
from src.models.large_cnn import LargeCNN
from src.models.small_cnn import SmallCNN


def build_model(name: str, in_channels: int, num_classes: int):
    name = name.lower()
    if name == "small":
        return SmallCNN(in_channels, num_classes)
    if name == "large":
        return LargeCNN(in_channels, num_classes)
    raise ValueError(f"Modelo desconocido: {name!r} (usa 'small' o 'large')")
