"""Orquestador de la CURVA DE DATOS: el experimento central de la Fase 1.

Entrena cada CONDICIÓN con cada FRACCIÓN de datos (y, si quieres, varias
seeds) y vuelca todos los resultados a un CSV. Con ese CSV luego dibujamos
la curva accuracy-vs-datos (make_figures.py).

La pregunta que responde: "¿cómo cae la accuracy de cada modelo según le
quitamos datos de entrenamiento?". La tesis predice que el modelo con el
módulo cortical aguanta mejor cuanto MENOS datos hay. De momento, sin el
módulo todavía construido, comparamos las dos referencias:

    A = David  (small, sin módulo)   <- el pequeño "pelado"
    C = Goliat (large, sin módulo)   <- el grande "pelado"

Cuando exista cortical.py añadiremos B (small+módulo, el héroe) y D
(large+módulo) a este mismo diccionario y el script los recorrerá igual.

Uso:
    # corrida completa (5 fracciones x 2 condiciones, seed 42)
    ./.venv/bin/python -m scripts.run_data_curve --epochs 10

    # prueba rápida para validar que el pipeline corre de punta a punta
    ./.venv/bin/python -m scripts.run_data_curve --quick
"""
import argparse
import csv
import os
import time

from src.train import TrainConfig, train_one

# Cada condición = una etiqueta legible + qué modelo usa.
# (Más adelante añadiremos "use_module": True/False cuando exista cortical.)
# Cada condición: etiqueta legible + modelo + si el frontal aprende sus pesos.
# Los controles A'/A'' comparten params con B para aislar la ESTRUCTURA.
CONDITIONS = {
    "A":   {"label": "David (small)",        "model": "small",          "learnable": False},
    "B":   {"label": "David+cortical",       "model": "small+cortical", "learnable": True},
    "Bf":  {"label": "David+cortical FIJO",  "model": "small+cortical", "learnable": False},
    "Bp":  {"label": "David+cortical PRETRAINED", "model": "small+cortical", "learnable": False, "pretrained": True},
    "Ap":  {"label": "A' control tamaño",    "model": "small+dumb",     "learnable": True},
    "App": {"label": "A'' lesión estructura","model": "small+lesion",   "learnable": True},
    "C":   {"label": "Goliat (large)",       "model": "large",          "learnable": False},
    "D":   {"label": "Goliat+cortical",      "model": "large+cortical", "learnable": True},
}

# Los puntos del eje X de la curva: porción del set de entrenamiento.
DEFAULT_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.0]


def run_curve(
    conditions,
    fractions,
    seeds,
    dataset: str,
    epochs: int,
    batch_size: int,
    lr: float,
    out_path: str,
    dict_path: str = None,
):
    """Recorre el producto condiciones x fracciones x seeds y guarda CSV."""
    rows = []
    total = len(conditions) * len(fractions) * len(seeds)
    i = 0
    t0 = time.time()

    for cond in conditions:
        spec = CONDITIONS[cond]
        for fraction in fractions:
            for seed in seeds:
                i += 1
                print(f"\n[{i}/{total}] condición {cond} ({spec['label']}) "
                      f"| fracción {fraction:.0%} | seed {seed}")

                # El Φ pre-entrenado solo se inyecta en condiciones "pretrained".
                cfg = TrainConfig(
                    model=spec["model"],
                    dataset=dataset,
                    fraction=fraction,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    learnable_cortical=spec.get("learnable", False),
                    dict_path=dict_path if spec.get("pretrained") else None,
                )
                # verbose=False: aquí no queremos el log época-a-época,
                # solo el resultado final de cada condición.
                result = train_one(cfg, verbose=False)

                # Añadimos columnas de identificación del experimento.
                result = {"condition": cond, "label": spec["label"], **result}
                rows.append(result)
                print(f"    -> test_acc {result['test_acc']:.4f} "
                      f"| {result['n_params']:,} params "
                      f"| {result['seconds']}s")

    _write_csv(rows, out_path)
    elapsed = time.time() - t0
    print(f"\nListo: {len(rows)} corridas en {elapsed:.0f}s. "
          f"Resultados -> {out_path}")
    return rows


def _write_csv(rows, out_path: str):
    """Vuelca la lista de dicts a CSV (una fila por corrida)."""
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Las claves de la primera fila definen las columnas (todas iguales).
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Curva de datos: accuracy vs cantidad de datos.")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--conditions", nargs="+", default=["A", "C"],
                   choices=list(CONDITIONS.keys()),
                   help="Qué condiciones correr (por defecto A y C).")
    p.add_argument("--fractions", nargs="+", type=float, default=None,
                   help="Fracciones del set de train (por defecto 0.01..1.0).")
    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="Una o varias seeds (varias = barras de error).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="results/data_curve.csv")
    p.add_argument("--dict_path", default=None,
                   help="Φ pre-entrenado (.pt) que recibe la condición Bp (pretrained).")
    p.add_argument("--quick", action="store_true",
                   help="Prueba mínima: 2 fracciones, 1 época. Valida el pipeline.")
    args = p.parse_args()

    fractions = args.fractions if args.fractions is not None else DEFAULT_FRACTIONS
    epochs = args.epochs
    if args.quick:
        fractions = [0.01, 0.05]
        epochs = 1
        print(">> modo --quick: 2 fracciones, 1 época (solo para validar el pipeline)")

    run_curve(
        conditions=args.conditions,
        fractions=fractions,
        seeds=args.seeds,
        dataset=args.dataset,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_path=args.out,
        dict_path=args.dict_path,
    )


if __name__ == "__main__":
    main()
