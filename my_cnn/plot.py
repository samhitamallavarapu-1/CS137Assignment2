import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path("/cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment2/my_cnn")
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def find_metrics_files(root_dir):
    return sorted(root_dir.glob("outputs_transfer_*/*_metrics.json"))


def parse_filename(file_path):
    """
    Example filename:
    resnet50_weather_pretrained_last_layer_metrics.json

    Returns:
    model_name = resnet50
    mode = last_layer
    """
    name = file_path.stem  # remove .json
    parts = name.split("_")

    model_name = parts[0]

    # training mode is always at the end before "metrics"
    mode = parts[-2]

    return model_name, mode


def load_metrics(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    epochs, rmse, auc = [], [], []
    test_metrics = None

    for entry in data:
        if entry["epoch"] == "test":
            test_metrics = entry
            continue

        epochs.append(entry["epoch"])
        rmse.append(entry.get("val_rmse_mean"))
        auc.append(entry.get("val_auc_apcp"))

    return epochs, rmse, auc, test_metrics


def plot_and_save(file_path):
    model_name, mode = parse_filename(file_path)

    model_dir = PLOTS_DIR / model_name
    model_dir.mkdir(exist_ok=True)

    epochs, rmse, auc, test_metrics = load_metrics(file_path)

    # =========================
    # RMSE plot
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, rmse, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE")
    plt.title(f"{model_name} - {mode} (RMSE)")
    plt.grid(True)

    if test_metrics:
        plt.figtext(
            0.15, 0.8,
            f"Test RMSE: {test_metrics['rmse_mean']:.3f}",
            bbox=dict(facecolor='white', alpha=0.8)
        )

    rmse_path = model_dir / f"{mode}_rmse.png"
    plt.savefig(rmse_path, dpi=300)
    plt.close()

    print(f"Saved RMSE -> {rmse_path}")

    # =========================
    # AUC plot
    # =========================
    if any(a is not None for a in auc):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, auc, marker='x', linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Validation AUC")
        plt.title(f"{model_name} - {mode} (AUC)")
        plt.grid(True)

        if test_metrics and "auc_apcp" in test_metrics:
            plt.figtext(
                0.15, 0.8,
                f"Test AUC: {test_metrics['auc_apcp']:.3f}",
                bbox=dict(facecolor='white', alpha=0.8)
            )

        auc_path = model_dir / f"{mode}_auc.png"
        plt.savefig(auc_path, dpi=300)
        plt.close()

        print(f"Saved AUC -> {auc_path}")


def main():
    files = find_metrics_files(ROOT_DIR)

    if not files:
        print("No metrics files found.")
        return

    print(f"Found {len(files)} files\n")

    for f in files:
        print(f"Processing: {f}")
        plot_and_save(f)


if __name__ == "__main__":
    main()