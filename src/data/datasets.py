"""Carga de datos y muestreo por fracciones (el eje "curva de datos").

Aquí vive todo lo relacionado con DATOS:
  - bajar el dataset (MNIST de momento) con torchvision
  - convertirlo a tensores normalizados
  - quedarnos con una FRACCIÓN del set de entrenamiento (1/5/10/25/100%)
    de forma estratificada y reproducible
  - envolverlo en DataLoaders listos para entrenar
"""
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.utils.seed import make_generator, seed_worker

# Estadísticas (media, std) de cada dataset, para normalizar.
# MNIST: 1 canal. CIFAR-10: 3 canales (RGB), medias/std estándar por canal.
_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

# Clase de torchvision por cada nombre de dataset.
_DATASETS = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
}

# Metadatos que los modelos necesitan para construirse genéricamente.
DATASET_META = {
    "mnist":   {"in_channels": 1, "num_classes": 10, "image_size": 32},  # 28->32 por el padding
    "cifar10": {"in_channels": 3, "num_classes": 10, "image_size": 32},  # ya nativo 32x32 RGB
}


def get_transform(name: str) -> transforms.Compose:
    """Pipeline de transformación de cada imagen antes de entrar a la red."""
    mean, std = _STATS[name]
    steps = []
    # MNIST/Fashion son 28x28. Las llevamos a 32x32 con padding de ceros
    # (un borde de 2 px). CIFAR-10 ya es 32x32 nativo, no necesita pad.
    # Sin data augmentation a propósito: aumentar datos confundiría el eje
    # "cuántos datos hacen falta" que mide la curva.
    if name in ("mnist", "fashion_mnist"):
        steps.append(transforms.Pad(2))            # 28x28 -> 32x32
    steps.append(transforms.ToTensor())            # PIL [0..255] -> tensor [0..1]
    steps.append(transforms.Normalize(mean, std))  # (x - mean) / std -> centrado en ~0
    return transforms.Compose(steps)


def get_dataset(name: str, train: bool, root: str = "data") -> Dataset:
    """Descarga (si hace falta) y devuelve el dataset de torchvision."""
    name = name.lower()
    ds_class = _DATASETS[name]
    return ds_class(root=root, train=train, download=True, transform=get_transform(name))


def make_fraction_subset(dataset: Dataset, fraction: float, seed: int) -> Dataset:
    """Submuestreo ESTRATIFICADO y reproducible del set de entrenamiento.

    Devuelve un subconjunto con `fraction` de los datos manteniendo la
    proporción de cada clase. Estratificar importa muchísimo con fracciones
    diminutas: si tomaras el 1% totalmente al azar, podrías quedarte casi
    sin ejemplos de alguna clase y arruinar la comparación.
    """
    if fraction >= 1.0:
        return dataset  # 100%: usamos el dataset entero

    targets = np.array(dataset.targets)
    rng = np.random.default_rng(seed)  # generador propio, sembrado

    selected = []
    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]                 # índices de esta clase
        n_keep = max(1, int(round(len(cls_idx) * fraction)))  # cuántos conservar
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        selected.append(chosen)

    indices = np.concatenate(selected)
    rng.shuffle(indices)  # mezclar para que no queden ordenados por clase
    return Subset(dataset, indices.tolist())


def get_dataloaders(
    name: str,
    fraction: float,
    batch_size: int,
    seed: int,
    root: str = "data",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Devuelve (train_loader, test_loader) reproducibles.

    El train_loader usa solo `fraction` de los datos; el test_loader usa
    SIEMPRE el set de test completo (para comparar todas las condiciones
    contra el mismo examen).
    """
    train_ds = get_dataset(name, train=True, root=root)
    train_ds = make_fraction_subset(train_ds, fraction, seed)
    test_ds = get_dataset(name, train=False, root=root)

    generator = make_generator(seed)  # controla el shuffle de forma reproducible
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,  # en test el orden da igual
        num_workers=num_workers,
    )
    return train_loader, test_loader
