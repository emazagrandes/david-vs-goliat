"""Entrenar UNA condición: un modelo + una fracción de datos + una seed.

Este es el motor. La función train_one(cfg) hace todo el ciclo de
aprendizaje y devuelve las métricas. Más adelante, run_data_curve.py
llamará a train_one muchas veces (cada condición × cada fracción × cada
seed) y juntará los resultados.

Uso desde terminal (ejemplo):
    ./.venv/bin/python -m src.train --model small --fraction 0.05 --epochs 10
"""
import argparse
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from src.data.datasets import DATASET_META, get_dataloaders
from src.eval import accuracy, count_parameters
from src.models import build_model
from src.utils.device import get_device
from src.utils.seed import set_seed


@dataclass
class TrainConfig:
    model: str = "small"      # "small" (David) o "large" (Goliat)
    dataset: str = "mnist"
    fraction: float = 1.0     # porción del set de entrenamiento (0..1)
    seed: int = 42
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3          # learning rate (tamaño del paso de aprendizaje)
    learnable_cortical: bool = False  # solo aplica a modelos "+cortical"
    dict_path: str = None     # Φ pre-entrenado para "+cortical" (None = aleatorio)


def train_one(cfg: TrainConfig, verbose: bool = True) -> dict:
    set_seed(cfg.seed)              # azar reproducible
    device = get_device()

    train_loader, test_loader = get_dataloaders(
        cfg.dataset, cfg.fraction, cfg.batch_size, cfg.seed
    )

    meta = DATASET_META[cfg.dataset]
    model = build_model(
        cfg.model, meta["in_channels"], meta["num_classes"],
        learnable_cortical=cfg.learnable_cortical,
        dict_path=cfg.dict_path,
    ).to(device)

    criterion = nn.CrossEntropyLoss()                       # función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # optimizador

    start = time.time()
    test_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()  # modo entrenamiento
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # --- los 5 pasos del aprendizaje ---
            optimizer.zero_grad()         # 1. borrar gradientes anteriores
            logits = model(x)             # 2. forward: predecir
            loss = criterion(logits, y)   # 3. medir el error
            loss.backward()               # 4. backward: calcular gradientes
            optimizer.step()              # 5. actualizar los pesos

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = accuracy(model, test_loader, device)
        if verbose:
            print(f"epoch {epoch:2d}/{cfg.epochs} | "
                  f"train_loss {train_loss:.4f} | test_acc {test_acc:.4f}")

    elapsed = time.time() - start
    n_params = count_parameters(model)
    return {
        **asdict(cfg),
        "test_acc": round(test_acc, 4),
        "n_params": n_params,
        "device": str(device),
        "seconds": round(elapsed, 1),
    }


def main():
    p = argparse.ArgumentParser(description="Entrenar una condición.")
    p.add_argument("--model", default="small",
                   choices=["small", "large",
                            "small+cortical", "large+cortical",
                            "small+dumb", "small+lesion",
                            "large+dumb", "large+lesion"])
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--learnable_cortical", action="store_true",
                   help="Deja que el diccionario del módulo aprenda (solo +cortical).")
    p.add_argument("--dict_path", default=None,
                   help="Ruta a un Φ pre-entrenado (.pt) para el módulo +cortical.")
    args = p.parse_args()

    result = train_one(TrainConfig(**vars(args)))
    print("RESULT:", result)


if __name__ == "__main__":
    main()
