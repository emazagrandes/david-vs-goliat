"""Dibuja la curva de datos a partir del CSV que produce run_data_curve.py.

Una figura = accuracy (eje Y) vs fracción de datos (eje X), una línea por
condición. El eje X va en escala logarítmica porque las fracciones
(1%, 5%, 10%, 25%, 100%) están muy juntas abajo y muy separadas arriba;
en log se ven repartidas y la forma de la curva se lee bien.

Si hay varias seeds por punto, dibuja la MEDIA y una banda de ±desviación
(las "barras de error": cuánto baila el resultado según el azar).

Uso:
    ./.venv/bin/python -m scripts.make_figures
    ./.venv/bin/python -m scripts.make_figures --csv results/data_curve.csv
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: str):
    """Lee el CSV y devuelve la lista de filas (cada una un dict)."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def aggregate(rows):
    """Agrupa por (condición, fracción) y promedia sobre seeds.

    Devuelve: {label: {"x": [fracciones], "mean": [...], "std": [...]}}
    ordenado por fracción ascendente, listo para plotear.
    """
    # acc_por_grupo[(label, fraction)] = [acc_seed1, acc_seed2, ...]
    acc_por_grupo = defaultdict(list)
    for r in rows:
        label = r["label"]
        fraction = float(r["fraction"])
        acc_por_grupo[(label, fraction)].append(float(r["test_acc"]))

    # reorganizamos por label
    out = defaultdict(lambda: {"x": [], "mean": [], "std": []})
    for (label, fraction), accs in sorted(acc_por_grupo.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out[label]["x"].append(fraction)
        out[label]["mean"].append(float(np.mean(accs)))
        out[label]["std"].append(float(np.std(accs)))
    return out


def plot_curve(data, out_path: str, dataset: str):
    """Dibuja una línea por condición con banda de ±std y guarda PNG."""
    plt.figure(figsize=(7, 5))
    for label, d in data.items():
        x = np.array(d["x"]) * 100      # a porcentaje para el eje
        mean = np.array(d["mean"])
        std = np.array(d["std"])
        plt.plot(x, mean, marker="o", label=label)
        # banda de error solo si hay dispersión (varias seeds)
        if std.any():
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xscale("log")
    plt.xlabel("% del set de entrenamiento (escala log)")
    plt.ylabel("Accuracy en test")
    plt.title(f"Curva de datos — {dataset.upper()}")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Figura guardada -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Dibuja la curva de datos desde el CSV.")
    p.add_argument("--csv", default="results/data_curve.csv")
    p.add_argument("--out", default="figures/data_curve.png")
    args = p.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        print(f"CSV vacío o inexistente: {args.csv}")
        return
    dataset = rows[0].get("dataset", "")
    data = aggregate(rows)
    plot_curve(data, args.out, dataset)


if __name__ == "__main__":
    main()
