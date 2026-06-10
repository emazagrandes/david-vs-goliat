"""Fábrica de modelos: traduce un nombre de condición a un modelo.

Nombres base: "small" (David) y "large" (Goliat).
Sufijo "+<frontal>": pone un preprocesador DELANTE de la CNN (opción A).
Frontales disponibles:
  - cortical  -> módulo cortical completo (sparse+predictive+divnorm)
  - lesion    -> control A'': MISMO módulo pero SIN recurrencia ni divnorm
  - dumb      -> control A': capa conv genérica, mismos params, cero estructura

Mapa de condiciones experimentales:
  - "small"          -> A   David pelado
  - "small+cortical" -> B   David + módulo (el héroe)
  - "small+dumb"     -> A'  control de tamaño (params equivalentes)
  - "small+lesion"   -> A'' control de estructura (módulo lesionado)
  - "large"          -> C   Goliat pelado
  - "large+cortical" -> D   Goliat + módulo
"""
import torch
import torch.nn as nn

from src.models.cortical import CorticalModule, DumbFrontEnd
from src.models.large_cnn import LargeCNN
from src.models.small_cnn import SmallCNN


def _build_backbone(base: str, in_channels: int, num_classes: int):
    """Construye solo la CNN (sin frontal)."""
    if base == "small":
        return SmallCNN(in_channels, num_classes)
    if base == "large":
        return LargeCNN(in_channels, num_classes)
    raise ValueError(f"Modelo desconocido: {base!r} (usa 'small' o 'large')")


def _load_pretrained(dict_path: str):
    """Carga (Φ, whiten) de un diccionario pre-entrenado."""
    payload = torch.load(dict_path, map_location="cpu")
    if isinstance(payload, dict):
        return payload["dictionary"], payload.get("whiten", False)
    return payload, False  # tensor pelado: sin metadatos


def _build_frontend(kind: str, in_channels: int, learnable: bool, dict_path: str = None):
    """Construye el preprocesador que va delante de la CNN.

    whiten=True en todas las variantes corticales: así B (learnable), Bf
    (random fijo) y Bp (pretrained fijo) comparten EXACTAMENTE la arquitectura
    y solo difieren en el origen de Φ. El whitening es fijo (0 params).
    """
    if kind == "cortical":          # módulo completo
        # Con dict_path -> Φ pre-entrenado (prior real); el whiten lo dicta el
        # propio diccionario guardado. Sin él -> Φ aleatorio, whiten ON igual.
        if dict_path:
            init_dict, whiten = _load_pretrained(dict_path)
        else:
            init_dict, whiten = None, True
        return CorticalModule(in_channels, learnable=learnable,
                              init_dict=init_dict, whiten=whiten)
    if kind == "lesion":            # A'': mismos filtros y params, SIN estructura
        return CorticalModule(in_channels, n_iters=1, use_divnorm=False,
                              learnable=learnable, whiten=True)
    if kind == "dumb":              # A': capa genérica con params equivalentes
        return DumbFrontEnd(in_channels)
    raise ValueError(f"Frontal desconocido: {kind!r} (usa cortical/lesion/dumb)")


def build_model(name: str, in_channels: int, num_classes: int,
                learnable_cortical: bool = False, dict_path: str = None):
    name = name.lower()
    if "+" in name:
        base, kind = name.split("+", 1)
        # El frontal recibe la imagen cruda y produce out_channels.
        frontend = _build_frontend(kind, in_channels, learnable_cortical, dict_path)
        # La CNN ahora ve los canales del CÓDIGO, no la imagen cruda.
        backbone = _build_backbone(base, frontend.out_channels, num_classes)
        # Sequential = frontal primero, CNN después (opción A: preprocesador).
        return nn.Sequential(frontend, backbone)
    return _build_backbone(name, in_channels, num_classes)
