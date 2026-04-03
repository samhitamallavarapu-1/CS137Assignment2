#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(".")
PLOT_DIR = ROOT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


VARIABLE_NAMES = [
    "TMP",
    "RH",
    "UGRD",
    "VGRD",
    "GUST",
    "APCP",
]


def find_weather_metrics_files(root_dir: Path):
    files = []
    for path in root_dir.rglob("*.json"):
        name = path.name.lower()
        parent = str(path.parent).lower()

        # Skip Stanford Cars / non-weather experiment folders
        if "cars" in name or "cars" in parent or "stanford_cars" in parent:
            continue

        if "metrics" in name or "training" in name or "history" in name:
            files.append(path)

    return sorted(files)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def is_weather_metrics(data):
    if not isinstance(data, list) or len(data) == 0:
        return False

    sample = data[0]
    weather_keys = {
        "val_rmse",
        "val_rmse_mean",
        "val_rmse_apcp_rain",
        "val_auc_apcp",
        "rmse",
        "rmse_mean",
        "rmse_apcp_rain",
        "auc_apcp",
    }

    return any(k in sample for k in weather_keys)


def extract_run_name(path: Path):
    """
    Build short, readable labels for plotting.
    """
    stem = path.stem.lower()
    parent = path.parent.name.lower()

    full_text = f"{parent}_{stem}"

    # Model name
    if "resnet" in full_text:
        model = "resnet"
    elif "densenet" in full_text:
        model = "densenet"
    elif "cnn" in full_text:
        model = "cnn"
    elif "weather" in full_text:
        model = "weather"
    else:
        model = "run"

    # Strategy
    if "scratch" in full_text:
        strategy = "scratch"
    elif "last_layer" in full_text or "last" in full_text:
        strategy = "last"
    elif "gradual" in full_text:
        strategy = "gradual"
    elif "full" in full_text:
        strategy = "full"
    elif "baseline" in full_text:
        strategy = "baseline"
    else:
        strategy = "default"

    return f"{model}\n{strategy}"


def get_last_epoch_row(metrics):
    epoch_rows = [m for m in metrics if m.get("epoch") != "test"]
    if not epoch_rows:
        return metrics[-1]
    return epoch_rows[-1]


def get_best_epoch_row(metrics):
    epoch_rows = [m for m in metrics if m.get("epoch") != "test"]
    if not epoch_rows:
        return metrics[-1]

    if "val_rmse_mean" in epoch_rows[0]:
        return min(epoch_rows, key=lambda x: x.get("val_rmse_mean", float("inf")))
    return epoch_rows[-1]


def get_test_row(metrics):
    for row in metrics:
        if row.get("epoch") == "test":
            return row
    return None


def get_metric_row(metrics, prefer="test"):
    if prefer == "test":
        row = get_test_row(metrics)
        if row is not None:
            return row

    if prefer == "best":
        return get_best_epoch_row(metrics)

    return get_last_epoch_row(metrics)


def get_rmse_vector(row):
    if "rmse" in row:
        return row["rmse"]
    if "val_rmse" in row:
        return row["val_rmse"]
    return None


def get_rmse_mean(row):
    if "rmse_mean" in row:
        return row["rmse_mean"]
    if "val_rmse_mean" in row:
        return row["val_rmse_mean"]

    rmse = get_rmse_vector(row)
    if rmse is not None:
        return float(np.mean(rmse))

    return None


def get_apcp_auc(row):
    if "auc_apcp" in row:
        return row["auc_apcp"]
    if "val_auc_apcp" in row:
        return row["val_auc_apcp"]
    return None


def get_apcp_rain_rmse(row):
    if "rmse_apcp_rain" in row:
        return row["rmse_apcp_rain"]
    if "val_rmse_apcp_rain" in row:
        return row["val_rmse_apcp_rain"]
    return None


def plot_rmse_by_variable(all_runs, prefer="test"):
    plt.figure(figsize=(12, 6))

    plotted = False
    for name, metrics in all_runs.items():
        row = get_metric_row(metrics, prefer=prefer)
        rmse = get_rmse_vector(row)
        if rmse is None or len(rmse) != len(VARIABLE_NAMES):
            continue

        plt.plot(VARIABLE_NAMES, rmse, marker="o", label=name)
        plotted = True

    if not plotted:
        print(f"No RMSE-by-variable data found for prefer={prefer}.")
        plt.close()
        return

    plt.ylabel("RMSE")
    plt.title(f"Weather RMSE per Variable ({prefer})")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"weather_rmse_per_variable_{prefer}.png", dpi=300)
    plt.close()


def plot_mean_rmse(all_runs, prefer="test"):
    names = []
    values = []

    for name, metrics in all_runs.items():
        row = get_metric_row(metrics, prefer=prefer)
        val = get_rmse_mean(row)
        if val is None:
            continue
        names.append(name)
        values.append(val)

    if not names:
        print(f"No mean RMSE data found for prefer={prefer}.")
        return

    plt.figure(figsize=(max(12, len(names) * 1.2), 5))
    plt.bar(names, values)
    plt.ylabel("Mean RMSE")
    plt.title(f"Weather Mean RMSE Comparison ({prefer})")
    plt.xticks(rotation=60, ha="right")
    plt.subplots_adjust(bottom=0.35)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"weather_mean_rmse_{prefer}.png", dpi=300)
    plt.close()


def plot_auc(all_runs, prefer="test"):
    names = []
    values = []

    for name, metrics in all_runs.items():
        row = get_metric_row(metrics, prefer=prefer)
        auc = get_apcp_auc(row)
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            continue
        names.append(name)
        values.append(auc)

    if not names:
        print(f"No APCP AUC data found for prefer={prefer}.")
        return

    plt.figure(figsize=(max(12, len(names) * 1.2), 5))
    plt.bar(names, values)
    plt.ylabel("AUC")
    plt.title(f"Weather APCP AUC Comparison ({prefer})")
    plt.xticks(rotation=60, ha="right")
    plt.subplots_adjust(bottom=0.35)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"weather_apcp_auc_{prefer}.png", dpi=300)
    plt.close()


def plot_apcp_rain_rmse(all_runs, prefer="test"):
    names = []
    values = []

    for name, metrics in all_runs.items():
        row = get_metric_row(metrics, prefer=prefer)
        val = get_apcp_rain_rmse(row)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        names.append(name)
        values.append(val)

    if not names:
        print(f"No APCP rain RMSE data found for prefer={prefer}.")
        return

    plt.figure(figsize=(max(12, len(names) * 1.2), 5))
    plt.bar(names, values)
    plt.ylabel("RMSE")
    plt.title(f"Weather APCP Rain RMSE ({prefer})")
    plt.xticks(rotation=60, ha="right")
    plt.subplots_adjust(bottom=0.35)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"weather_apcp_rain_rmse_{prefer}.png", dpi=300)
    plt.close()


def plot_training_curves(all_runs):
    plt.figure(figsize=(12, 6))

    plotted = False
    for name, metrics in all_runs.items():
        epoch_rows = [m for m in metrics if m.get("epoch") != "test"]
        if not epoch_rows:
            continue

        epochs = [m["epoch"] for m in epoch_rows]
        vals = [get_rmse_mean(m) for m in epoch_rows]

        if all(v is None for v in vals):
            continue

        plt.plot(epochs, vals, marker="o", label=name)
        plotted = True

    if not plotted:
        print("No training-curve data found.")
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel("Validation Mean RMSE")
    plt.title("Weather Training Curves")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "weather_training_curves.png", dpi=300)
    plt.close()


def main():
    json_files = find_weather_metrics_files(ROOT_DIR)

    if not json_files:
        print("No candidate JSON files found.")
        return

    print("Found candidate JSON files:")
    for f in json_files:
        print(" ", f)

    all_runs = {}
    for f in json_files:
        try:
            data = load_json(f)
        except Exception as e:
            print(f"Skipping {f} (load error: {e})")
            continue

        if not is_weather_metrics(data):
            print(f"Skipping {f} (not weather metrics)")
            continue

        run_name = extract_run_name(f)
        all_runs[run_name] = data

    if not all_runs:
        print("No weather metrics files found after filtering.")
        return

    print("\nUsing weather runs:")
    for name in all_runs:
        print(" ", name)

    # Test-based comparisons
    plot_rmse_by_variable(all_runs, prefer="test")
    plot_mean_rmse(all_runs, prefer="test")
    plot_auc(all_runs, prefer="test")
    plot_apcp_rain_rmse(all_runs, prefer="test")

    # Best-validation comparisons
    plot_rmse_by_variable(all_runs, prefer="best")
    plot_mean_rmse(all_runs, prefer="best")
    plot_auc(all_runs, prefer="best")
    plot_apcp_rain_rmse(all_runs, prefer="best")

    # Learning curves
    plot_training_curves(all_runs)

    print(f"\nSaved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()