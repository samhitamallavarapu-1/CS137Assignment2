import torch
import torch.nn as nn
from pathlib import Path


class WeatherCNN(nn.Module):
    """
    Simple CNN for 24-hour weather forecasting at the Jumbo grid point.

    Expected raw input shape:
        (B, H, W, C)

    Output shape:
        (B, 6)
    """

    def __init__(
        self,
        n_channels: int,
        n_outputs: int = 6,
        normalize_inside_model: bool = False,
    ):
        super().__init__()

        self.normalize_inside_model = normalize_inside_model

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

        if self.normalize_inside_model:
            x = (x - self.input_mean.view(1, 1, 1, -1)) / self.input_std.view(1, 1, 1, -1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.encoder(x)
        x = self.head(x)
        return x


def _load_normalization_stats(model: WeatherCNN, model_dir: Path) -> None:
    norm_path = model_dir / "normalization_stats.pt"
    if not norm_path.exists():
        return

    stats = torch.load(norm_path, map_location="cpu")
    input_mean = stats["input_mean"].float()
    input_std = torch.clamp(stats["input_std"].float(), min=1e-6)

    model.input_mean.copy_(input_mean)
    model.input_std.copy_(input_std)


def _load_weights(model: WeatherCNN, model_dir: Path) -> None:
    weights_path = model_dir / "my_cnn_weights.pth"
    if not weights_path.exists():
        return

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)


def get_model(metadata: dict, for_evaluation: bool = True) -> WeatherCNN:
    """
    for_evaluation=True:
        model expects raw inputs and normalizes internally

    for_evaluation=False:
        model expects already-normalized inputs
    """
    n_channels = int(metadata["n_vars"])
    model = WeatherCNN(
        n_channels=n_channels,
        n_outputs=6,
        normalize_inside_model=for_evaluation,
    )

    model_dir = Path(__file__).resolve().parent
    _load_normalization_stats(model, model_dir)
    _load_weights(model, model_dir)

    model.eval()
    return model