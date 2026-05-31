"""Goliat: una CNN grande (~cientos de miles de parámetros).

Es el baseline fuerte (condición C). Misma estructura conceptual que David
pero con más canales, una convolución extra y una capa oculta en el
clasificador -> mucha más capacidad. Mismo interfaz que SmallCNN.
"""
import torch
import torch.nn as nn


class LargeCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # convolución extra
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # capa oculta (David no la tiene)
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
