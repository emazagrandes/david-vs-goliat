"""Reproducibilidad: fijar todas las fuentes de aleatoriedad.

Un experimento científico debe poder repetirse y dar el MISMO resultado.
En deep learning hay varias fuentes de azar (inicialización de pesos,
barajado de datos, dropout...). Si no las fijamos, dos ejecuciones del
mismo código dan números distintos y no sabríamos si una mejora viene
del módulo cortical o simplemente de "otra tirada de dados".
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija la semilla en las tres librerías que generan azar."""
    random.seed(seed)            # módulo random de Python (p.ej. shuffles)
    np.random.seed(seed)         # NumPy (transformaciones, sampling)
    torch.manual_seed(seed)      # PyTorch en CPU (init de pesos, dropout)
    torch.cuda.manual_seed_all(seed)  # PyTorch en GPU (si hay)

    # cuDNN: forzar algoritmos deterministas a costa de algo de velocidad.
    # Sin esto, la misma convolución puede dar resultados ligeramente
    # distintos entre ejecuciones.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Reseed de cada worker del DataLoader.

    El DataLoader puede usar varios procesos en paralelo para cargar datos.
    Cada proceso hereda una semilla distinta; esto la re-fija para que el
    orden de los datos sea reproducible aunque haya varios workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Generador con semilla para pasar al DataLoader (controla el shuffle)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
