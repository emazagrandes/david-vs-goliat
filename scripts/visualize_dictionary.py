"""Visualizar los átomos del diccionario Φ como imágenes — el control de
sanidad clásico de Olshausen-Field.

Si el pre-entrenamiento (sparse coding no supervisado) funcionó, los átomos
φ_i deben parecerse a filtros de Gabor: bordes localizados y orientados a
varias escalas, igual que los campos receptivos de la corteza visual V1.
Verlos a ojo confirma que Φ es un PRIOR de imágenes naturales y no ruido.

Cada átomo tiene forma [in_channels, k, k] (p.ej. [3, 7, 7] en CIFAR). Para
mostrarlo lo escalamos a [0,1] de forma independiente (min-max por átomo),
porque lo que importa es el PATRÓN, no la magnitud absoluta.

Uso:
    ./.venv/bin/python -m scripts.visualize_dictionary \
        --dict assets/dict_cifar10_32x7.pt \
        --out figures/dictionary_cifar10.png
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")  # backend sin ventana, solo guarda a archivo
import matplotlib.pyplot as plt
import torch


def _to_displayable(atom: torch.Tensor) -> "np.ndarray":
    """Escala un átomo a [0,1] (min-max) y lo deja en formato imagen.

    [C, k, k] -> si C==3 lo tratamos como RGB (k,k,3); si C==1 escala de
    grises (k,k). El min-max es POR ÁTOMO: realza el patrón sin que un
    átomo de baja amplitud se vea negro al lado de otro fuerte.
    """
    a = atom.detach().cpu().float()
    a = a - a.min()
    a = a / (a.max() + 1e-8)            # ahora en [0,1]
    if a.shape[0] == 3:                 # RGB
        return a.permute(1, 2, 0).numpy()   # [C,k,k] -> [k,k,C]
    return a[0].numpy()                 # 1 canal -> [k,k]


def visualize(dict_path: str, out_path: str):
    payload = torch.load(dict_path, map_location="cpu")
    dictionary = payload["dictionary"] if isinstance(payload, dict) else payload
    n_atoms = dictionary.shape[0]
    dataset = payload.get("dataset", "?") if isinstance(payload, dict) else "?"

    # Rejilla cuadrada (32 átomos -> 6x6, sobran celdas que dejamos vacías).
    cols = math.ceil(math.sqrt(n_atoms))
    rows = math.ceil(n_atoms / cols)
    is_rgb = dictionary.shape[1] == 3

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    axes = axes.flatten()
    for i in range(rows * cols):
        axes[i].axis("off")
        if i < n_atoms:
            img = _to_displayable(dictionary[i])
            axes[i].imshow(img, cmap=None if is_rgb else "gray",
                           interpolation="nearest")

    fig.suptitle(f"Diccionario Φ — {dataset} · {n_atoms} átomos "
                 f"{dictionary.shape[2]}x{dictionary.shape[3]}"
                 f"{' RGB' if is_rgb else ' (gris)'}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figura guardada -> {out_path}  ({n_atoms} átomos, "
          f"{'RGB' if is_rgb else 'gris'})")


def main():
    p = argparse.ArgumentParser(description="Visualizar átomos del diccionario.")
    p.add_argument("--dict", required=True, help="Ruta al .pt del diccionario.")
    p.add_argument("--out", default=None, help="PNG de salida (por defecto junto al .pt).")
    args = p.parse_args()

    out = args.out or os.path.join(
        "figures", os.path.basename(args.dict).replace(".pt", ".png"))
    visualize(args.dict, out)


if __name__ == "__main__":
    main()
