import torch
import torch.nn as nn


class WeatherCNN(nn.Module):
    """CNN for spatial weather forecasting."""

    def __init__(self, n_channels: int, n_outputs: int = 6):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, 450, 449, c)
        x = x.permute(0, 3, 1, 2)  # (B, c, H, W)
        x = self.encoder(x)
        x = self.head(x)
        return x


def get_model(metadata: dict) -> WeatherCNN:
    """Build model from metadata."""
    n_channels = int(metadata.get("n_vars", len(metadata.get("variable_names", []))))
    return WeatherCNN(n_channels=n_channels, n_outputs=6)
