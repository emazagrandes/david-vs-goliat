"""Métricas y evaluación: la "vara de medir" común a todo el proyecto.

Aquí viven las funciones que cuantifican un modelo. Las usan tanto
`train.py` (durante el entrenamiento) como el orquestador de la curva de
datos. Tener UNA sola definición de cada métrica evita que dos sitios
midan "accuracy" de formas ligeramente distintas y las cifras dejen de
ser comparables.

Tres familias de métricas:
  - CALIDAD     -> accuracy (¿cuántas acierta?)
  - TAMAÑO      -> count_parameters (¿cuántos pesos tiene?)
  - CÓMPUTO     -> count_flops (¿cuántas operaciones por imagen?)

El eje de cómputo es clave para la honestidad del proyecto: si el módulo
cortical ayuda a David pero le añade muchísimo cómputo, la comparación
"David alcanza a Goliat" hay que matizarla. Por eso medimos y reportamos
el coste, no solo la accuracy.

Uso desde terminal:
    ./.venv/bin/python -m src.eval
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.data.datasets import DATASET_META
from src.models import build_model
from src.utils.device import get_device


@torch.no_grad()  # evaluar no necesita gradientes -> más rápido y menos memoria
def accuracy(model: nn.Module, loader, device) -> float:
    """Fracción de aciertos sobre el loader dado (0..1).

    Es LA función de accuracy del proyecto: train.py y el orquestador la
    importan de aquí, así nadie la reimplementa distinto.
    """
    model.eval()  # modo evaluación: desactiva dropout, fija batchnorm, etc.
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)          # clase con mayor logit
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Número de pesos del modelo (el "tamaño" de David o Goliat).

    Un parámetro es cada número aprendible: cada peso de cada convolución
    y de cada capa lineal, más los sesgos (bias). `trainable_only` cuenta
    solo los que se actualizan (requires_grad=True); si el módulo cortical
    fuera de pesos FIJOS, esos no contarían como entrenables.
    """
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def count_flops(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    device: Optional[torch.device] = None,  # aceptado por compatibilidad; no se usa
) -> Tuple[int, int]:
    """Coste de cómputo por UNA imagen: devuelve (flops, macs).

    - MAC = Multiply-ACcumulate, una multiplicación + una suma (a*b + c).
      Es la operación básica de convoluciones y capas lineales.
    - FLOP = FLoating-point OPeration. Por convención 1 MAC = 2 FLOPs
      (la multiplicación y la suma cuentan como dos operaciones).

    Medimos sobre un batch de tamaño 1 (una sola imagen) para tener el
    coste "por muestra", independiente del batch_size.

    Dos detalles importantes:
      1. El conteo se hace en CPU: el número de operaciones es el mismo en
         cualquier hardware, y así evitamos cualquier rareza de MPS.
      2. thop "ensucia" el modelo al que perfila (le pega buffers float64
         total_ops/total_params, que además MPS no soporta). Por eso
         perfilamos una COPIA y dejamos intacto el modelo original.
    """
    import copy

    from thop import profile  # import local: thop solo se usa aquí

    clone = copy.deepcopy(model).to("cpu")  # copia desechable
    clone.eval()
    dummy = torch.randn(1, *input_shape)    # (batch=1, canales, alto, ancho)
    macs, _ = profile(clone, inputs=(dummy,), verbose=False)
    macs = int(macs)
    flops = macs * 2
    return flops, macs


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    device,
    input_shape: Tuple[int, int, int],
) -> dict:
    """Ficha completa de un modelo: calidad + tamaño + cómputo en un dict.

    Reúne las tres familias de métricas de una vez para que el orquestador
    pueda volcarlas a una tabla/CSV sin llamar a cada función por separado.
    """
    acc = accuracy(model, test_loader, device)
    n_params = count_parameters(model)
    flops, macs = count_flops(model, input_shape, device=device)
    return {
        "test_acc": round(acc, 4),
        "n_params": n_params,
        "flops": flops,
        "macs": macs,
    }


@torch.no_grad()
def evaluate_ood(model: nn.Module, ood_loader, device) -> float:
    """Accuracy sobre UN loader OUT-OF-DISTRIBUTION (corrupto/desplazado)."""
    return accuracy(model, ood_loader, device)


@torch.no_grad()
def evaluate_ood_suite(
    model: nn.Module,
    device,
    root: str,
    name: str = "cifar10",
    corruptions=None,
    severity: int = None,
    batch_size: int = 256,
) -> dict:
    """Accuracy en CADA corrupción de CIFAR-10-C + la media.

    Devuelve {"ood_<corrupción>": acc, ..., "ood_mean": media}. El desglose
    por corrupción permite ver si la divisive normalization ayuda
    ESPECÍFICAMENTE donde la teoría predice (contrast, brightness), no solo
    en media.
    """
    from src.data.datasets import CIFAR10C_CORRUPTIONS, get_ood_loader

    corrs = corruptions or CIFAR10C_CORRUPTIONS
    out = {}
    for c in corrs:
        loader = get_ood_loader(c, batch_size, root=root, name=name,
                                severity=severity)
        out[f"ood_{c}"] = round(accuracy(model, loader, device), 4)
    out["ood_mean"] = round(sum(out.values()) / len(out), 4)
    return out


def _human(n: int) -> str:
    """Formatea un número grande de forma legible: 619786 -> '619.8K'."""
    for unit in ("", "K", "M", "G"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n}"
        n /= 1000.0
    return f"{n:.1f}T"


def main():
    """Compara coste de David vs Goliat (sin entrenar): params y FLOPs."""
    device = get_device()
    meta = DATASET_META["mnist"]
    # forma de entrada de UNA imagen MNIST tras el padding: (canales, 32, 32)
    input_shape = (meta["in_channels"], meta["image_size"], meta["image_size"])

    print(f"{'modelo':<8} {'params':>12} {'FLOPs/img':>14} {'MACs/img':>14}")
    print("-" * 50)
    for name in ("small", "large"):
        model = build_model(name, meta["in_channels"], meta["num_classes"])
        n_params = count_parameters(model)
        flops, macs = count_flops(model, input_shape, device=device)
        etiqueta = "David" if name == "small" else "Goliat"
        print(f"{etiqueta:<8} {n_params:>12,} "
              f"{_human(flops):>14} {_human(macs):>14}")


if __name__ == "__main__":
    main()
