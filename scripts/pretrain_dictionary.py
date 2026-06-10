"""Pre-entrenar el DICCIONARIO del módulo cortical (sparse coding NO supervisado).

El test honesto del "prior gratis" de Olshausen-Field necesita un diccionario
Φ que capture la estadística de imágenes naturales (átomos tipo Gabor / bordes),
NO una proyección aleatoria. Este script aprende ese Φ resolviendo el problema
clásico de DICTIONARY LEARNING:

    min_{Φ, a}  ½‖x − Φᵀa‖²₂ + λ‖a‖₁     s.a.  ‖φ_i‖₂ = 1

de forma alternada, sobre imágenes de entrenamiento SIN sus etiquetas:

  1. INFERENCIA (código): con Φ fijo, inferir el código disperso a con el
     mismo bucle ISTA del módulo (CorticalModule.forward).
  2. ACTUALIZACIÓN (diccionario): con a fijo, un paso de gradiente sobre Φ
     que reduce el error de reconstrucción ‖x − Φᵀa‖².
  3. NORMALIZACIÓN: re-escalar cada átomo a norma unidad (evita que Φ crezca
     sin límite para abaratar el término de sparsity).

El resultado es un Φ que, congelado en B-fijo (init_dict + learnable=False),
es el inductive bias cortical de verdad, sin coste de parámetros entrenables
ni etiquetas.

Uso:
    # diccionario para CIFAR-10 (3 canales), 32 átomos 7x7
    ./.venv/bin/python -m scripts.pretrain_dictionary --dataset cifar10 \
        --out assets/dict_cifar10_32x7.pt

    # prueba rápida (pocas imágenes, pocas épocas) para validar la mecánica
    ./.venv/bin/python -m scripts.pretrain_dictionary --dataset cifar10 --quick
"""
import argparse
import os

import torch
import torch.nn.functional as F

from src.data.datasets import DATASET_META, get_dataloaders
from src.models.cortical import CorticalModule
from src.utils.device import get_device
from src.utils.seed import set_seed


@torch.no_grad()
def _normalize_atoms(dictionary: torch.Tensor) -> None:
    """Re-escala in-place cada átomo a norma L2 = 1 (restricción ‖φ_i‖₂=1)."""
    norms = dictionary.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    dictionary.div_(norms.clamp_min(1e-8))


def pretrain(
    dataset: str,
    n_atoms: int,
    kernel_size: int,
    n_iters: int,
    lambda_sparse: float,
    step_size: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    out_path: str,
    whiten: bool = True,
    max_batches: int = None,
):
    """Aprende Φ por dictionary learning no supervisado y lo guarda en disco."""
    set_seed(seed)
    device = get_device()
    in_channels = DATASET_META[dataset]["in_channels"]

    # Usamos el train set ENTERO (fraction=1.0) pero IGNORAMOS las etiquetas:
    # el pre-entrenamiento es no supervisado, no consume "datos etiquetados".
    train_loader, _ = get_dataloaders(dataset, 1.0, batch_size, seed)

    # Módulo APRENDIBLE: aquí sí dejamos que Φ tenga gradiente (es lo que
    # entrenamos). Tras esto se congela en B-fijo.
    #
    # use_divnorm=False A PROPÓSITO: la normalización divisiva es una
    # no-linealidad de control de ganancia que ROMPE la interpretación
    # a = argmin ½‖x−Φa‖² + λ‖a‖₁. Para el dictionary learning necesitamos
    # que 'a' sea el código de sparse coding REAL (ISTA puro), si no el paso
    # de actualización de Φ no minimiza el modelo generativo x≈Φa y diverge.
    # La divnorm se reactiva luego en el experimento (módulo completo).
    module = CorticalModule(
        in_channels, n_atoms=n_atoms, kernel_size=kernel_size,
        n_iters=n_iters, lambda_sparse=lambda_sparse, step_size=step_size,
        use_divnorm=False, learnable=True, whiten=whiten,
    ).to(device)
    _normalize_atoms(module.dictionary.data)

    optimizer = torch.optim.Adam([module.dictionary], lr=lr)
    padding = kernel_size // 2

    print(f">> pre-entrenando Φ: {dataset} | {n_atoms} átomos {kernel_size}x{kernel_size} "
          f"x {in_channels}ch | whiten={whiten} | device {device}")

    for epoch in range(1, epochs + 1):
        running_recon = 0.0
        running_sparsity = 0.0
        n_seen = 0
        for b, (x, _) in enumerate(train_loader):   # '_' = etiquetas IGNORADAS
            if max_batches is not None and b >= max_batches:
                break
            x = x.to(device)

            # 1. INFERENCIA del código disperso a con Φ actual (sin gradiente:
            #    en este paso Φ se trata como fijo, como en ISTA alternado).
            #    Si hay whitening, el código representa la imagen BLANQUEADA,
            #    así que ese es también el objetivo de reconstrucción.
            with torch.no_grad():
                a = module(x)
                target = module._apply_whitening(x) if module.whiten else x

            # 2. ACTUALIZACIÓN de Φ: gradiente del error de reconstrucción.
            recon = F.conv_transpose2d(a, module.dictionary, padding=padding)
            recon_loss = F.mse_loss(recon, target)
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            # 3. NORMALIZACIÓN: cada átomo vuelve a norma unidad.
            _normalize_atoms(module.dictionary.data)

            running_recon += recon_loss.item() * x.size(0)
            running_sparsity += (a == 0).float().mean().item() * x.size(0)
            n_seen += x.size(0)

        print(f"epoch {epoch:2d}/{epochs} | recon_mse {running_recon / n_seen:.5f} "
              f"| sparsity {running_sparsity / n_seen:.1%}")

    dictionary = module.dictionary.detach().cpu()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "dictionary": dictionary,
            "dataset": dataset,
            "n_atoms": n_atoms,
            "kernel_size": kernel_size,
            "in_channels": in_channels,
            "lambda_sparse": lambda_sparse,
            "whiten": whiten,
        },
        out_path,
    )
    print(f"\nListo. Φ guardado -> {out_path}  (forma {tuple(dictionary.shape)})")
    return dictionary


def main():
    p = argparse.ArgumentParser(description="Pre-entrenar el diccionario cortical (no supervisado).")
    p.add_argument("--dataset", default="cifar10", choices=list(DATASET_META.keys()))
    p.add_argument("--n_atoms", type=int, default=32)
    p.add_argument("--kernel_size", type=int, default=7)
    p.add_argument("--n_iters", type=int, default=10,
                   help="Iteraciones ISTA para inferir el código (más que en train).")
    p.add_argument("--lambda_sparse", type=float, default=0.02,
                   help="Con whitening la señal es ~0.2 de escala; λ≈0.02 da sparsity ~95%.")
    p.add_argument("--step_size", type=float, default=0.1,
                   help="Paso ISTA. Con divnorm OFF debe ser pequeño (η≤1/L) o diverge.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="assets/dict_cifar10_32x7.pt")
    p.add_argument("--no_whiten", dest="whiten", action="store_false",
                   help="Desactiva el whitening (por defecto ON, da átomos Gabor).")
    p.set_defaults(whiten=True)
    p.add_argument("--quick", action="store_true",
                   help="Prueba mínima: 3 épocas, 20 batches. Valida la mecánica.")
    args = p.parse_args()

    epochs, max_batches = args.epochs, None
    if args.quick:
        epochs, max_batches = 3, 20
        print(">> modo --quick: 3 épocas x 20 batches (solo valida la mecánica)")

    pretrain(
        dataset=args.dataset,
        n_atoms=args.n_atoms,
        kernel_size=args.kernel_size,
        n_iters=args.n_iters,
        lambda_sparse=args.lambda_sparse,
        step_size=args.step_size,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        out_path=args.out,
        whiten=args.whiten,
        max_batches=max_batches,
    )


if __name__ == "__main__":
    main()
