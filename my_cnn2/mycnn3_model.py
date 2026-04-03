import torch
import torch.nn as nn
from pathlib import Path


class WeatherCNN(nn.Module):
    """
    CNN for 24-hour weather forecasting.

    Evaluator input:
        x shape = (B, H, W, C)

    Output:
        (B, 6)
    """

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

        self.register_buffer("input_mean", torch.zeros(n_channels))
        self.register_buffer("input_std", torch.ones(n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, H, W, C), got {tuple(x.shape)}")

        # Normalize raw evaluator inputs inside the model
        x = (x - self.input_mean.view(1, 1, 1, -1)) / self.input_std.view(1, 1, 1, -1)

        # channels-last -> channels-first
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.encoder(x)
        x = self.head(x)

        # IMPORTANT: do NOT unnormalize outputs
        return x


def _load_input_normalization_stats(model: WeatherCNN, model_dir: Path) -> None:
    norm_path = model_dir / "normalization_stats.pt"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalization stats file: {norm_path}")

    stats = torch.load(norm_path, map_location="cpu")
    input_mean = stats["input_mean"].float()
    input_std = torch.clamp(stats["input_std"].float(), min=1e-6)

    model.input_mean.copy_(input_mean)
    model.input_std.copy_(input_std)


def _load_weights(model: WeatherCNN, model_dir: Path) -> None:
    weights_path = model_dir / "my_cnn3_weights.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint)
        return

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return

    raise ValueError(f"Unsupported checkpoint format in {weights_path}")


def build_model(metadata: dict) -> WeatherCNN:
    n_channels = int(metadata["n_vars"])
    model = WeatherCNN(n_channels=n_channels, n_outputs=6)
    return model


def get_model(metadata: dict) -> WeatherCNN:
    n_channels = int(metadata["n_vars"])
    model = WeatherCNN(n_channels=n_channels, n_outputs=6)

    model_dir = Path(__file__).resolve().parent
    _load_input_normalization_stats(model, model_dir)
    _load_weights(model, model_dir)

    model.eval()
    return model