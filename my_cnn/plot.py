import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path("/cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment2/my_cnn")
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def find_metrics_files(root_dir):
    return sorted(root_dir.rglob("*_metrics.json"))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def safe_mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def detect_dataset_type(file_path):
    """
    Cars:
      folder or filename contains '_cars_'
    Weather:
      everything else
    """
    path_str = str(file_path).lower()
    if "_cars_" in path_str:
        return "cars"
    return "weather"


def detect_format(data):
    """
    Detect JSON schema from keys in the first non-test row.
    """
    if not isinstance(data, list) or len(data) == 0:
        return "unknown"

    sample = None
    for entry in data:
        if entry.get("epoch") != "test":
            sample = entry
            break

    if sample is None:
        sample = data[0]

    # Cars / classification
    if any(k in sample for k in ["val_accuracy_top1", "val_accuracy_top5"]):
        return "classification"

    # Weather / regression
    if any(
        k in sample
        for k in [
            "val_rmse_mean",
            "val_rmse",
            "val_rmse_apcp_rain",
            "val_auc_apcp",
            "rmse",
            "rmse_apcp_rain",
            "auc_apcp",
        ]
    ):
        return "weather"

    return "unknown"


def canonicalize_strategy(strategy, strategy_note=None):
    if not strategy:
        return "unknown"

    s = strategy.strip().lower()
    note = (strategy_note or "").strip().lower()

    if s in {"full", "scratch", "frozen", "last_layer"}:
        return s

    if "all parameters trainable" in note:
        return "full"
    if "only last layer trainable" in note or "only final layer trainable" in note or "last layer" in note:
        return "last_layer"
    if "feature extractor frozen" in note or "frozen" in note:
        return "frozen"
    if "scratch" in note:
        return "scratch"

    return s.replace(" ", "_")


def infer_mode_from_filename(file_path, default="run"):
    path_str = str(file_path).lower()
    stem = file_path.stem.lower()

    tokens = stem.replace("-", "_").split("_")
    path_tokens = path_str.replace("-", "_").split("_")
    all_tokens = tokens + path_tokens

    if "scratch" in all_tokens:
        return "scratch"
    if ("last" in all_tokens and "layer" in all_tokens) or "lastlayer" in all_tokens:
        return "last_layer"
    if "frozen" in all_tokens:
        return "frozen"
    if "full" in all_tokens:
        return "full"
    if "pretrained" in all_tokens:
        return "pretrained"

    return default


def infer_model_name(file_path):
    name = file_path.stem
    parts = name.split("_")
    return parts[0] if parts else "model"


def parse_run_identity(file_path, data):
    dataset_type = detect_dataset_type(file_path)
    model_name = infer_model_name(file_path)

    strategy = None
    strategy_note = None
    for entry in data:
        if entry.get("epoch") != "test":
            strategy = entry.get("strategy")
            strategy_note = entry.get("strategy_note")
            break

    if strategy is not None:
        mode = canonicalize_strategy(strategy, strategy_note)
    else:
        mode = infer_mode_from_filename(file_path, default=f"{dataset_type}_run")

    return dataset_type, model_name, mode


def add_metrics_box(lines, loc="upper_right"):
    if not lines:
        return

    ax = plt.gca()

    if loc == "upper_left":
        x, y, ha, va = 0.02, 0.98, "left", "top"
    elif loc == "lower_right":
        x, y, ha, va = 0.98, 0.02, "right", "bottom"
    elif loc == "lower_left":
        x, y, ha, va = 0.02, 0.02, "left", "bottom"
    else:
        x, y, ha, va = 0.98, 0.98, "right", "top"

    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )


# =========================================================
# WEATHER METRICS
# =========================================================
def load_weather_metrics(data):
    epochs = []
    train_loss = []
    val_loss = []
    val_rmse_mean = []
    val_rmse_apcp_rain = []
    val_auc_apcp = []
    test_metrics = None

    for entry in data:
        epoch = entry.get("epoch")

        if epoch == "test":
            test_metrics = entry
            continue

        epochs.append(epoch)
        train_loss.append(entry.get("train_loss"))
        val_loss.append(entry.get("val_loss"))
        val_rmse_mean.append(entry.get("val_rmse_mean"))

        if "val_rmse_apcp_rain" in entry and entry["val_rmse_apcp_rain"] is not None:
            rain_rmse = entry["val_rmse_apcp_rain"]
        elif "rmse_apcp_rain" in entry and entry["rmse_apcp_rain"] is not None:
            rain_rmse = entry["rmse_apcp_rain"]
        else:
            rain_rmse = None

        if "val_auc_apcp" in entry and entry["val_auc_apcp"] is not None:
            auc_val = entry["val_auc_apcp"]
        elif "auc_apcp" in entry and entry["auc_apcp"] is not None:
            auc_val = entry["auc_apcp"]
        else:
            auc_val = None

        val_rmse_apcp_rain.append(rain_rmse)
        val_auc_apcp.append(auc_val)

    return (
        epochs,
        train_loss,
        val_loss,
        val_rmse_mean,
        val_rmse_apcp_rain,
        val_auc_apcp,
        test_metrics,
    )


def get_weather_test_loss(test_metrics):
    if not test_metrics:
        return None
    return test_metrics.get("loss")


def get_weather_test_rmse_mean(test_metrics):
    if not test_metrics:
        return None

    if "rmse_mean" in test_metrics and test_metrics["rmse_mean"] is not None:
        return test_metrics["rmse_mean"]
    if "val_rmse_mean" in test_metrics and test_metrics["val_rmse_mean"] is not None:
        return test_metrics["val_rmse_mean"]
    if "rmse" in test_metrics and isinstance(test_metrics["rmse"], list):
        return safe_mean(test_metrics["rmse"])

    return None


def get_weather_test_rmse_apcp_rain(test_metrics):
    if not test_metrics:
        return None

    if "rmse_apcp_rain" in test_metrics and test_metrics["rmse_apcp_rain"] is not None:
        return test_metrics["rmse_apcp_rain"]
    if "val_rmse_apcp_rain" in test_metrics and test_metrics["val_rmse_apcp_rain"] is not None:
        return test_metrics["val_rmse_apcp_rain"]

    return None


def get_weather_test_auc(test_metrics):
    if not test_metrics:
        return None

    if "auc_apcp" in test_metrics and test_metrics["auc_apcp"] is not None:
        return test_metrics["auc_apcp"]
    if "val_auc_apcp" in test_metrics and test_metrics["val_auc_apcp"] is not None:
        return test_metrics["val_auc_apcp"]

    return None


def plot_weather(file_path, model_dir, dataset_type, model_name, mode, data):
    (
        epochs,
        train_loss,
        val_loss,
        val_rmse_mean,
        val_rmse_apcp_rain,
        val_auc_apcp,
        test_metrics,
    ) = load_weather_metrics(data)

    # Loss plot
    valid_train_loss = [(e, v) for e, v in zip(epochs, train_loss) if v is not None]
    valid_val_loss = [(e, v) for e, v in zip(epochs, val_loss) if v is not None]

    if valid_train_loss or valid_val_loss:
        plt.figure(figsize=(8, 5))

        if valid_train_loss:
            x, y = zip(*valid_train_loss)
            plt.plot(x, y, marker="o", label="Train Loss")

        if valid_val_loss:
            x, y = zip(*valid_val_loss)
            plt.plot(x, y, marker="s", label="Val Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_type.upper()} | {model_name} | {mode} | Loss")
        plt.grid(True)
        plt.legend(loc="upper right")

        test_loss = get_weather_test_loss(test_metrics)
        if test_loss is not None:
            add_metrics_box([f"Test Loss: {test_loss:.3f}"], loc="upper_left")

        out_path = model_dir / f"{dataset_type}_{mode}_loss.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Loss -> {out_path}")
    else:
        print(f"No weather loss values found in {file_path}")

    # RMSE mean plot
    valid_rmse_mean = [(e, v) for e, v in zip(epochs, val_rmse_mean) if v is not None]
    if valid_rmse_mean:
        plt.figure(figsize=(8, 5))
        x, y = zip(*valid_rmse_mean)
        plt.plot(x, y, marker="o", label="Val RMSE Mean")

        plt.xlabel("Epoch")
        plt.ylabel("RMSE Mean")
        plt.title(f"{dataset_type.upper()} | {model_name} | {mode} | RMSE Mean")
        plt.grid(True)
        plt.legend(loc="upper right")

        test_rmse_mean = get_weather_test_rmse_mean(test_metrics)
        if test_rmse_mean is not None:
            add_metrics_box([f"Test RMSE Mean: {test_rmse_mean:.3f}"], loc="upper_left")

        out_path = model_dir / f"{dataset_type}_{mode}_rmse_mean.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved RMSE Mean -> {out_path}")
    else:
        print(f"No weather RMSE mean values found in {file_path}")

    # Rain RMSE plot
    valid_rmse_rain = [(e, v) for e, v in zip(epochs, val_rmse_apcp_rain) if v is not None]
    if valid_rmse_rain:
        plt.figure(figsize=(8, 5))
        x, y = zip(*valid_rmse_rain)
        plt.plot(x, y, marker="o", label="Val RMSE APCP Rain")

        plt.xlabel("Epoch")
        plt.ylabel("RMSE APCP Rain")
        plt.title(f"{dataset_type.upper()} | {model_name} | {mode} | RMSE APCP Rain")
        plt.grid(True)
        plt.legend(loc="upper right")

        test_rmse_rain = get_weather_test_rmse_apcp_rain(test_metrics)
        if test_rmse_rain is not None:
            add_metrics_box([f"Test RMSE APCP Rain: {test_rmse_rain:.3f}"], loc="upper_left")

        out_path = model_dir / f"{dataset_type}_{mode}_rmse_apcp_rain.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved RMSE APCP Rain -> {out_path}")
    else:
        print(f"No weather rain-RMSE values found in {file_path}")

    # AUC plot
    valid_auc = [(e, v) for e, v in zip(epochs, val_auc_apcp) if v is not None]
    if valid_auc:
        plt.figure(figsize=(8, 5))
        x, y = zip(*valid_auc)
        plt.plot(x, y, marker="x", linestyle="--", label="Val AUC APCP")

        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"{dataset_type.upper()} | {model_name} | {mode} | AUC APCP")
        plt.grid(True)
        plt.legend(loc="lower right")

        test_auc = get_weather_test_auc(test_metrics)
        if test_auc is not None:
            add_metrics_box([f"Test AUC APCP: {test_auc:.3f}"], loc="upper_left")

        out_path = model_dir / f"{dataset_type}_{mode}_auc_apcp.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved AUC APCP -> {out_path}")
    else:
        print(f"No weather AUC values found in {file_path}")


# =========================================================
# CLASSIFICATION METRICS
# =========================================================
def load_classification_metrics(data):
    epochs = []
    train_loss = []
    val_loss = []
    train_top1 = []
    val_top1 = []
    val_top5 = []
    test_metrics = None
    strategy = None

    for entry in data:
        epoch = entry.get("epoch")

        if epoch == "test":
            test_metrics = entry
            continue

        epochs.append(epoch)
        train_loss.append(entry.get("train_loss"))
        val_loss.append(entry.get("val_loss"))
        train_top1.append(entry.get("train_accuracy_top1"))
        val_top1.append(entry.get("val_accuracy_top1"))
        val_top5.append(entry.get("val_accuracy_top5"))

        if strategy is None:
            strategy = entry.get("strategy")

    return epochs, train_loss, val_loss, train_top1, val_top1, val_top5, test_metrics, strategy


def plot_classification(file_path, model_dir, dataset_type, model_name, mode, data):
    (
        epochs,
        train_loss,
        val_loss,
        train_top1,
        val_top1,
        val_top5,
        test_metrics,
        strategy,
    ) = load_classification_metrics(data)

    title_mode = mode
    if strategy and strategy.lower() != mode.lower():
        title_mode = f"{mode} ({strategy})"

    # Loss plot
    valid_train_loss = [(e, v) for e, v in zip(epochs, train_loss) if v is not None]
    valid_val_loss = [(e, v) for e, v in zip(epochs, val_loss) if v is not None]

    if valid_train_loss or valid_val_loss:
        plt.figure(figsize=(8, 5))

        if valid_train_loss:
            x, y = zip(*valid_train_loss)
            plt.plot(x, y, marker="o", label="Train Loss")

        if valid_val_loss:
            x, y = zip(*valid_val_loss)
            plt.plot(x, y, marker="s", label="Val Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_type.upper()} | {model_name} | {title_mode} | Loss")
        plt.grid(True)
        plt.legend(loc="upper right")

        note_lines = []
        if test_metrics and "loss" in test_metrics:
            note_lines.append(f"Test Loss: {test_metrics['loss']:.3f}")

        if note_lines:
            add_metrics_box(note_lines, loc="upper_left")

        out_path = model_dir / f"{dataset_type}_{mode}_loss.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Loss -> {out_path}")
    else:
        print(f"No classification loss values found in {file_path}")

    # Accuracy plot
    valid_train_top1 = [(e, v) for e, v in zip(epochs, train_top1) if v is not None]
    valid_val_top1 = [(e, v) for e, v in zip(epochs, val_top1) if v is not None]
    valid_val_top5 = [(e, v) for e, v in zip(epochs, val_top5) if v is not None]

    if valid_train_top1 or valid_val_top1 or valid_val_top5:
        plt.figure(figsize=(8, 5))

        if valid_train_top1:
            x, y = zip(*valid_train_top1)
            plt.plot(x, y, marker="o", label="Train Top-1 Acc")

        if valid_val_top1:
            x, y = zip(*valid_val_top1)
            plt.plot(x, y, marker="x", linestyle="--", label="Val Top-1 Acc")

        if valid_val_top5:
            x, y = zip(*valid_val_top5)
            plt.plot(x, y, marker="^", linestyle=":", label="Val Top-5 Acc")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset_type.upper()} | {model_name} | {title_mode} | Accuracy")
        plt.grid(True)
        plt.legend(loc="upper left")

        note_lines = []
        if test_metrics:
            if "accuracy_top1" in test_metrics:
                note_lines.append(f"Test Top-1: {test_metrics['accuracy_top1']:.3f}")
            if "accuracy_top5" in test_metrics:
                note_lines.append(f"Test Top-5: {test_metrics['accuracy_top5']:.3f}")

        if note_lines:
            add_metrics_box(note_lines, loc="upper_right")

        out_path = model_dir / f"{dataset_type}_{mode}_accuracy.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Accuracy -> {out_path}")
    else:
        print(f"No classification accuracy values found in {file_path}")


def plot_and_save(file_path):
    data = load_json(file_path)

    dataset_type, model_name, mode = parse_run_identity(file_path, data)
    model_dir = PLOTS_DIR / dataset_type / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    fmt = detect_format(data)

    print(f"  dataset={dataset_type} | model={model_name} | mode={mode} | format={fmt}")

    if fmt == "weather":
        plot_weather(file_path, model_dir, dataset_type, model_name, mode, data)
    elif fmt == "classification":
        plot_classification(file_path, model_dir, dataset_type, model_name, mode, data)
    else:
        print(f"Skipping unknown JSON format: {file_path}")


def main():
    files = find_metrics_files(ROOT_DIR)

    if not files:
        print("No metrics files found.")
        return

    print(f"Found {len(files)} metrics files.\n")

    for file_path in files:
        print(f"Processing: {file_path}")
        plot_and_save(file_path)
        print()

    print("Done.")


if __name__ == "__main__":
    main()