import json
from pathlib import Path

import matplotlib.pyplot as plt
import math


METRICS_PATH = Path("outputs/training_metrics.json")
PLOTS_DIR = Path("outputs/plots")


def load_metrics(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("training_metrics.json is empty or malformed")

    return data


def get_series(metrics, key):
    xs = []
    ys = []

    for row in metrics:
        if row.get("epoch") == "test":
            continue

        y = row.get(key)
        if y is None:
            continue

        if isinstance(y, float) and math.isnan(y):
            continue

        xs.append(row["epoch"])
        ys.append(y)

    return xs, ys


def save_plot(x, y, title, ylabel, output_path):
    if len(x) == 0 or len(y) == 0:
        print(f"Skipping empty plot: {output_path.name}")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot -> {output_path}")


def main():
    metrics = load_metrics(METRICS_PATH)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train loss
    x, y = get_series(metrics, "train_loss")
    save_plot(
        x, y,
        title="Training Loss by Epoch",
        ylabel="Train Loss",
        output_path=PLOTS_DIR / "train_loss.png",
    )

    # 2. Validation RMSE mean
    x, y = get_series(metrics, "val_rmse_mean")
    save_plot(
        x, y,
        title="Validation RMSE Mean by Epoch",
        ylabel="Validation RMSE Mean",
        output_path=PLOTS_DIR / "val_rmse_mean.png",
    )

    # 3. Validation APCP rainy RMSE
    x, y = get_series(metrics, "val_rmse_apcp_rain")
    save_plot(
        x, y,
        title="Validation Rainy-Sample APCP RMSE by Epoch",
        ylabel="Rainy APCP RMSE",
        output_path=PLOTS_DIR / "val_rmse_apcp_rain.png",
    )

    # 4. Validation APCP AUC
    x, y = get_series(metrics, "val_auc_apcp")
    save_plot(
        x, y,
        title="Validation APCP AUC by Epoch",
        ylabel="APCP AUC",
        output_path=PLOTS_DIR / "val_auc_apcp.png",
    )


if __name__ == "__main__":
    main()