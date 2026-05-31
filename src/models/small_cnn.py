"""David: una CNN deliberadamente pequeña (~pocos miles de parámetros).

Es el baseline débil (condición A) y, cuando le enchufemos el módulo
cortical, el héroe (condición B). Mismo interfaz que LargeCNN para poder
intercambiarlos sin tocar el resto del código.
"""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        # "features": extrae características de la imagen con convoluciones.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # detecta bordes simples
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # reduce a la mitad (28->14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),           # combina en patrones
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # (14->7)
        )
        # Pooling adaptativo: deja el mapa en 4x4 sea cual sea el tamaño de
        # entrada (28x28 de MNIST o 32x32 de CIFAR) -> el clasificador no
        # depende del dataset.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        # "classifier": de las características a las 10 clases.
        self.classifier = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)   # (batch, 32, 4, 4) -> (batch, 512)
        return self.classifier(x)  # logits, uno por clase
