"""Módulo cortical: el corazón de la tesis (Fase 2).

Un microcircuito inspirado en la corteza visual que se coloca DELANTE de la
CNN (sobre los píxeles crudos, "opción A") y transforma la imagen en un
CÓDIGO DISPERSO antes de que la red la vea.

Unifica tres principios de neurociencia computacional en UN solo bucle
recurrente (Lian & Burkitt 2025):

  1. SPARSE CODING (Olshausen & Field 1996)
     Representar la imagen con pocas "neuronas" activas a la vez; casi todo
     el código es cero.  -> lo da el umbral (soft-threshold).

  2. PREDICTIVE CODING (Rao & Ballard 1999)
     El circuito intenta RECONSTRUIR su entrada y solo el ERROR de predicción
     (lo que no supo reconstruir) impulsa la siguiente iteración.
     -> lo da el bucle  error = x - reconstrucción.

  3. DIVISIVE NORMALIZATION (Heeger 1992)
     Cada neurona se divide por la actividad de sus vecinas: control de
     ganancia / contraste.  -> lo da _divisive_norm().

El bucle es un ISTA convolucional (Iterative Shrinkage-Thresholding): el
algoritmo clásico para resolver sparse coding, "desenrollado" en T pasos.
Cada paso ES predictive coding; el umbral da la sparsity; la norma divisiva
mete el control de ganancia. Tres ideas, un circuito.

Pesos FIJOS (inductive bias puro, 0 params entrenables) vs APRENDIDOS
(end-to-end): lo controla el flag `learnable`. Los flags `n_iters` y
`use_divnorm` permiten las ABLACIONES (módulo "lesionado") que necesita el
control de confounds: si quitar la recurrencia o la normalización borra la
ventaja, es prueba de que la estructura importaba.

Uso desde terminal (prueba rápida de mecánica):
    ./.venv/bin/python -m src.models.cortical
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorticalModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_atoms: int = 32,
        kernel_size: int = 7,
        n_iters: int = 5,
        lambda_sparse: float = 0.1,
        step_size: float = 0.5,
        use_divnorm: bool = True,
        norm_sigma: float = 1e-2,
        learnable: bool = False,
        init_dict: torch.Tensor = None,
    ):
        super().__init__()
        self.n_iters = n_iters
        self.lambda_sparse = lambda_sparse
        self.step_size = step_size
        self.use_divnorm = use_divnorm
        self.norm_sigma = norm_sigma
        self.out_channels = n_atoms        # nº de canales que recibirá la CNN
        self.padding = kernel_size // 2    # 'same': preserva alto x ancho

        # El DICCIONARIO D: n_atoms "plantillas" (átomos) de tamaño kxk.
        # Forma [n_atoms, in_channels, k, k]. Se usa en los dos sentidos:
        #   - codificar:    conv2d(x, D)            imagen -> espacio de código
        #   - reconstruir:  conv_transpose2d(a, D)  código -> imagen
        # Compartir D en ambos sentidos ("weight tying") es lo estándar en
        # sparse coding: el mismo átomo que detecta también reconstruye.
        # (La forma [n_atoms, in_ch, k, k] sirve tal cual para conv2d Y para
        #  conv_transpose2d, por cómo PyTorch ordena los pesos.)
        #
        # init_dict: si se pasa un diccionario PRE-ENTRENADO (sparse coding no
        # supervisado, atoms tipo Olshausen-Field), se usa como inicialización.
        # Con learnable=False queda CONGELADO -> prior real de imágenes
        # naturales, no proyección aleatoria.
        if init_dict is not None:
            expected = (n_atoms, in_channels, kernel_size, kernel_size)
            if tuple(init_dict.shape) != expected:
                raise ValueError(
                    f"init_dict tiene forma {tuple(init_dict.shape)}, "
                    f"se esperaba {expected}")
            weight = init_dict.clone().float()
        else:
            weight = torch.randn(n_atoms, in_channels, kernel_size, kernel_size)
            norms = weight.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            weight = weight / norms
        self.dictionary = nn.Parameter(weight, requires_grad=learnable)

    def _soft_threshold(self, u: torch.Tensor) -> torch.Tensor:
        """Shrinkage no negativo: relu(u - lambda).

        Empuja a CERO todo lo que no supere el umbral lambda. Es lo que
        produce la SPARSITY: solo sobreviven las activaciones fuertes.
        """
        return F.relu(u - self.lambda_sparse)

    def _divisive_norm(self, a: torch.Tensor) -> torch.Tensor:
        """Cada canal dividido por la actividad RMS entre canales (mismo pixel).

        DIVISIVE NORMALIZATION / control de ganancia: si en un punto muchas
        neuronas disparan fuerte, todas se atenúan. Estabiliza el código y,
        en teoría, ayuda frente a cambios de contraste/brillo (OOD).
        """
        power = a.pow(2).mean(dim=1, keepdim=True)          # [B,1,H,W]
        return a / torch.sqrt(power + self.norm_sigma ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # a = código disperso que buscamos. Empieza en cero (sin actividad).
        b, _, h, w = x.shape
        a = x.new_zeros(b, self.out_channels, h, w)

        for _ in range(self.n_iters):
            recon = F.conv_transpose2d(a, self.dictionary, padding=self.padding)  # x̂: predicción de x
            error = x - recon                                                     # e: error de predicción
            grad = F.conv2d(error, self.dictionary, padding=self.padding)         # e -> espacio de código
            a = self._soft_threshold(a + self.step_size * grad)                   # paso + sparsity
            if self.use_divnorm:
                a = self._divisive_norm(a)                                        # control de ganancia
        return a


class DumbFrontEnd(nn.Module):
    """Control A': una capa convolucional GENÉRICA delante de la CNN.

    Misma forma de salida que el módulo cortical (n_atoms canales) y el
    MISMO número de parámetros (kernel kxk, sin bias), pero CERO estructura:
    no hay recurrencia, ni error de predicción, ni sparsity, ni norma
    divisiva. Es solo `relu(conv(x))`.

    Responde a la pregunta del confound de tamaño: ¿la ventaja de B viene de
    la estructura cortical, o de simplemente haber añadido una capa con más
    parámetros? Si B no le gana a esto, era músculo, no astucia.
    """

    def __init__(self, in_channels: int, out_channels: int = 32, kernel_size: int = 7):
        super().__init__()
        # bias=False para igualar EXACTAMENTE los params del diccionario
        # cortical (n_atoms x in_channels x k x k, sin término de sesgo).
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))


def main():
    """Prueba de mecánica: shapes, sparsity y flujo de gradiente."""
    torch.manual_seed(0)
    x = torch.randn(4, 1, 32, 32)  # 4 imágenes tipo MNIST (ya con padding a 32)

    print("== módulo FIJO (inductive bias puro) ==")
    mod = CorticalModule(in_channels=1, n_atoms=32, learnable=False)
    a = mod(x)
    sparsity = (a == 0).float().mean().item()
    n_train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    print(f"  entrada:  {tuple(x.shape)}")
    print(f"  código:   {tuple(a.shape)}   (canales 1 -> {mod.out_channels})")
    print(f"  sparsity: {sparsity:.1%} de ceros")
    print(f"  params entrenables: {n_train}")

    print("\n== módulo APRENDIDO (end-to-end) ==")
    mod_l = CorticalModule(in_channels=1, n_atoms=32, learnable=True)
    a_l = mod_l(x)
    loss = a_l.sum()
    loss.backward()  # ¿fluye el gradiente hasta el diccionario?
    grad_ok = mod_l.dictionary.grad is not None
    n_train_l = sum(p.numel() for p in mod_l.parameters() if p.requires_grad)
    print(f"  params entrenables: {n_train_l}")
    print(f"  gradiente llega al diccionario: {grad_ok}")


if __name__ == "__main__":
    main()
