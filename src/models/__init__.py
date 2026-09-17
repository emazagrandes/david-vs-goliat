"""Fábrica de modelos: traduce un nombre de condición a un modelo.

Nombres base: "small" (David) y "large" (Goliat).
Sufijo "+<frontal>": pone un módulo-PREFIJO delante de la CNN. Desde el
resultado en CIFAR (la "opción A" que REEMPLAZA la imagen perdía info y
perjudicaba), el prefijo ya NO reemplaza: CONCATENA su código con la imagen
cruda (opción "augment"). Así el CNN ve TODO (píxeles + features corticales)
y el módulo solo puede ayudar o ser ignorado, nunca destruir.

Frontales disponibles:
  - cortical  -> módulo cortical (sparse+predictive+divnorm), Φ fijo o aprendido
  - lesion    -> control A'': MISMO módulo pero SIN recurrencia ni divnorm
  - dumb      -> control A': capa conv genérica, mismos params, cero estructura

Mapa de condiciones:
  small / large            -> A / C   backbones pelados (sin prefijo)
  small+cortical (fijo)    -> Bf/Bp   David + módulo (Φ aleatorio o pretrained)
  small+cortical (aprend.) -> B       David + módulo con Φ entrenable
  small+dumb               -> A'      control de capacidad
  small+lesion             -> A''     control de estructura
  large+cortical           -> D       Goliat + módulo
"""
import torch
import torch.nn as nn

from src.models.cortical import CorticalModule, DumbFrontEnd
from src.models.large_cnn import LargeCNN
from src.models.small_cnn import SmallCNN

# Nº de átomos del diccionario / canales del código. 256 = SOBRE-COMPLETO
# (un parche es 7x7x3=147 dim; 256 > 147). Coates & Ng 2011: en CIFAR el
# número de features es la palanca nº1. dumb y cortical lo comparten para
# que A' siga emparejado en params con B.
N_ATOMS = 256


class ConcatFrontEnd(nn.Module):
    """Prefijo que AUMENTA en vez de reemplazar: concatena [imagen ; código].

    El backbone recibe los canales crudos de la imagen MÁS los canales del
    código del módulo. Propiedad clave: el CNN puede ignorar el código y
    actuar como el backbone pelado, así que el prefijo nunca puede empeorar
    el rendimiento por debajo del baseline; solo añadir señal útil.
    """

    def __init__(self, frontend: nn.Module, in_channels: int):
        super().__init__()
        self.frontend = frontend
        self.out_channels = in_channels + frontend.out_channels  # raw + código

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.frontend(x)
        return torch.cat([x, a], dim=1)


def _build_backbone(base: str, in_channels: int, num_classes: int):
    """Construye solo la CNN (sin prefijo)."""
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
    """Construye el módulo-prefijo (antes de concatenar con la imagen).

    SIN whitening en despliegue: el blanqueo borra el color/baja frecuencia,
    que CIFAR necesita. El color ya viaja en la imagen cruda concatenada; aquí
    queremos que el código aporte features, no que destruya color.
    """
    if kind == "cortical":          # módulo completo
        # Con dict_path -> Φ pre-entrenado (prior real). Sin él -> Φ aleatorio.
        # whiten lo dicta el diccionario guardado (lo pretrenamos sin whitening).
        if dict_path:
            init_dict, whiten = _load_pretrained(dict_path)
            n_atoms = init_dict.shape[0]
        else:
            init_dict, whiten, n_atoms = None, False, N_ATOMS
        return CorticalModule(in_channels, n_atoms=n_atoms, learnable=learnable,
                              init_dict=init_dict, whiten=whiten)
    if kind == "lesion":            # A'': mismos params, SIN estructura
        return CorticalModule(in_channels, n_atoms=N_ATOMS, n_iters=1,
                              use_divnorm=False, learnable=learnable, whiten=False)
    if kind == "dumb":              # A': capa genérica, params equivalentes
        return DumbFrontEnd(in_channels, out_channels=N_ATOMS)
    raise ValueError(f"Frontal desconocido: {kind!r} (usa cortical/lesion/dumb)")


def build_model(name: str, in_channels: int, num_classes: int,
                learnable_cortical: bool = False, dict_path: str = None):
    name = name.lower()
    if "+" in name:
        base, kind = name.split("+", 1)
        frontend = _build_frontend(kind, in_channels, learnable_cortical, dict_path)
        # AUMENTAR: el prefijo concatena su código con la imagen cruda.
        prefix = ConcatFrontEnd(frontend, in_channels)
        # El backbone ve raw + código (in_channels + n_atoms).
        backbone = _build_backbone(base, prefix.out_channels, num_classes)
        return nn.Sequential(prefix, backbone)
    return _build_backbone(name, in_channels, num_classes)
