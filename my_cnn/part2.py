import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import get_model
from analysis import compute_input_saliency


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def choose_indices(times, year):
    years = times.astype("datetime64[Y]").astype(int) + 1970
    idx = np.where(years == year)[0]
    return idx[idx + 24 < len(times)]


def load_valid_indices(dataset_dir, times, index_list, target_values, binary_labels):
    valid = []

    for t_idx in tqdm(index_list, desc="Filtering invalid/missing samples"):
        if t_idx + 24 >= len(times):
            continue

        dt = pd.Timestamp(times[t_idx])
        path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

        if not path.exists():
            continue

        try:
            x = torch.load(path, weights_only=True).float()
        except Exception:
            continue

        y = target_values[t_idx + 24]
        b = binary_labels[t_idx + 24].float()

        if not torch.isfinite(x).all():
            continue
        if not torch.isfinite(y).all():
            continue
        if not torch.isfinite(b).all():
            continue

        valid.append(int(t_idx))

    return np.array(valid, dtype=int)


def get_validation_indices(dataset_dir, times, target_values, binary_labels, seed, rescan_nan, cache_dir):
    all_years = list(range(2018, 2025))
    candidate_all = []

    for year in all_years:
        candidate_all.extend(choose_indices(times, year))

    candidate_all = np.array(candidate_all, dtype=int)

    cache_train_val_path = cache_dir / "valid_indices_train_val.npy"
    cache_test_path = cache_dir / "valid_indices_test.npy"

    if cache_train_val_path.exists() and cache_test_path.exists() and not rescan_nan:
        log("Loading valid indices from cache...")
        valid_train_val = np.load(cache_train_val_path)
        valid_test = np.load(cache_test_path)
    else:
        log("Computing valid indices...")
        valid_all = load_valid_indices(dataset_dir, times, candidate_all, target_values, binary_labels)

        if len(valid_all) < 2:
            raise RuntimeError("Not enough valid samples after filtering")

        rng = np.random.default_rng(seed)
        shuffled_all = rng.permutation(valid_all)

        split_idx = int(len(shuffled_all) * 0.8)
        valid_train_val = shuffled_all[:split_idx]
        valid_test = shuffled_all[split_idx:]

        np.save(cache_train_val_path, valid_train_val)
        np.save(cache_test_path, valid_test)
        log(f"Saved valid-index caches to {cache_dir}")

    rng = np.random.default_rng(seed)
    shuffled_train_val = rng.permutation(valid_train_val)

    n_train_val = len(shuffled_train_val)
    n_train = int(n_train_val * 0.8)

    train_idxs = shuffled_train_val[:n_train]
    val_idxs = shuffled_train_val[n_train:]

    log(f"Train samples: {len(train_idxs)}")
    log(f"Val samples: {len(val_idxs)}")
    log(f"Test samples: {len(valid_test)}")

    return train_idxs, val_idxs, valid_test


def load_normalization_stats(norm_path: Path, n_channels: int):
    if norm_path.exists():
        log(f"Loading normalization stats from {norm_path}")
        stats = torch.load(norm_path, weights_only=False)
        input_mean = stats["input_mean"].float()
        input_std = torch.clamp(stats["input_std"].float(), min=1e-6)
    else:
        log(f"Normalization stats not found at {norm_path}; using identity normalization.")
        input_mean = torch.zeros(n_channels)
        input_std = torch.ones(n_channels)

    return input_mean, input_std


def evaluate_validation_examples(
    model,
    dataset_dir,
    times,
    target_values,
    val_idxs,
    device,
    input_mean,
    input_std,
    apcp_idx=5,
):
    model.eval()
    criterion = nn.MSELoss(reduction="mean")

    examples = []

    input_mean = input_mean.to(device)
    input_std = input_std.to(device)

    with torch.no_grad():
        for t_idx in tqdm(val_idxs, desc="Evaluating validation set"):
            if t_idx + 24 >= len(times):
                continue

            dt = pd.Timestamp(times[t_idx])
            path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

            try:
                x_raw = torch.load(path, weights_only=True).float()
            except Exception:
                continue

            y = target_values[t_idx + 24].float().cpu()

            if not torch.isfinite(x_raw).all():
                continue
            if not torch.isfinite(y).all():
                continue

            x_norm = (x_raw.to(device) - input_mean.view(1, 1, -1)) / input_std.view(1, 1, -1)

            if not torch.isfinite(x_norm).all():
                continue

            pred = model(x_norm.unsqueeze(0)).squeeze(0).cpu()

            if not torch.isfinite(pred).all():
                continue

            full_loss = float(criterion(pred, y).item())
            apcp_loss = float((pred[apcp_idx] - y[apcp_idx]).pow(2).item())

            true_label = int(y[apcp_idx].item() > 2.0)
            pred_label = int(pred[apcp_idx].item() > 2.0)
            correct = (true_label == pred_label)

            confidence = abs(pred[apcp_idx].item() - 2.0)

            examples.append({
                "t_idx": int(t_idx),
                "timestamp": str(dt),
                "input_path": str(path),
                "full_loss": full_loss,
                "apcp_loss": apcp_loss,
                "true_apcp_value": float(y[apcp_idx].item()),
                "pred_apcp_value": float(pred[apcp_idx].item()),
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": bool(correct),
                "confidence": float(confidence),
                "target_vector": y.tolist(),
                "pred_vector": pred.tolist(),
            })

    if len(examples) == 0:
        raise RuntimeError("No valid validation examples found.")

    overall_accuracy = float(np.mean([ex["correct"] for ex in examples]))
    return examples, overall_accuracy


def save_feature_saliency(
    model,
    weights_path,
    input_tensor_norm,
    feature_names,
    output_dir,
    example_prefix,
    device,
    target_index=5,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_input_saliency(
        model=model,
        weights_path=weights_path,
        input_tensor=input_tensor_norm.unsqueeze(0),
        target_index=target_index,
        device=device,
        apply_abs=True,
        reduce_batch=True,
    )

    feature_scores = result["feature_scores_mean"].numpy()
    saliency_df = pd.DataFrame({
        "feature": feature_names,
        "saliency": feature_scores,
    }).sort_values("saliency", ascending=False)

    csv_path = output_dir / f"{example_prefix}_feature_saliency.csv"
    saliency_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(saliency_df)), saliency_df["saliency"].values)
    plt.xticks(range(len(saliency_df)), saliency_df["feature"].values, rotation=90)
    plt.ylabel("Mean absolute saliency")
    plt.title(f"{example_prefix} feature saliency")
    plt.tight_layout()

    png_path = output_dir / f"{example_prefix}_feature_saliency.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    torch.save(result["saliency"], output_dir / f"{example_prefix}_full_saliency.pt")

    return {
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "saliency_tensor_path": str(output_dir / f"{example_prefix}_full_saliency.pt"),
        "top_features": saliency_df.head(10).to_dict(orient="records"),
    }


def select_confident_examples(examples, k=3):
    correct_examples = [ex for ex in examples if ex["correct"]]
    incorrect_examples = [ex for ex in examples if not ex["correct"]]

    correct_examples = sorted(correct_examples, key=lambda x: x["confidence"], reverse=True)
    incorrect_examples = sorted(incorrect_examples, key=lambda x: x["confidence"], reverse=True)

    return correct_examples[:k], incorrect_examples[:k]


def main():
    parser = argparse.ArgumentParser(description="Find confidently correct/incorrect validation examples and save saliency.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"),
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        required=True,
        help="Path to trained model .pth",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path("normalization_stats.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("confidence_analysis_outputs"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rescan-nan", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--apcp-index", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_root
    metadata = torch.load(dataset_dir / "metadata.pt", weights_only=False)
    targets_data = torch.load(dataset_dir / "targets.pt", weights_only=False)

    times = targets_data["time"]
    target_values = targets_data["values"]
    binary_labels = targets_data["binary_label"]

    variable_names = metadata.get("variable_names", [f"feature_{i}" for i in range(int(metadata["n_vars"]))])
    n_channels = int(metadata["n_vars"])

    train_idxs, val_idxs, _ = get_validation_indices(
        dataset_dir=dataset_dir,
        times=times,
        target_values=target_values,
        binary_labels=binary_labels,
        seed=args.seed,
        rescan_nan=args.rescan_nan,
        cache_dir=cache_dir,
    )

    input_mean, input_std = load_normalization_stats(args.normalization_stats, n_channels)

    device = torch.device(args.device)

    model = get_model(metadata).to(device)
    state_dict = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    examples, overall_accuracy = evaluate_validation_examples(
        model=model,
        dataset_dir=dataset_dir,
        times=times,
        target_values=target_values,
        val_idxs=val_idxs,
        device=device,
        input_mean=input_mean,
        input_std=input_std,
        apcp_idx=args.apcp_index,
    )

    log(f"Validation APCP accuracy: {overall_accuracy:.4f}")

    top_correct, top_incorrect = select_confident_examples(examples, k=args.top_k)

    selected = {
        "confidently_correct": top_correct,
        "confidently_incorrect": top_incorrect,
    }

    summary = {
        "weights_path": str(args.weights_path),
        "normalization_stats": str(args.normalization_stats),
        "validation_accuracy_apcp": overall_accuracy,
        "n_validation_examples": len(examples),
        "confidently_correct": [],
        "confidently_incorrect": [],
    }

    input_mean_cpu = input_mean.cpu()
    input_std_cpu = input_std.cpu()

    for group_name, group_examples in selected.items():
        group_dir = args.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for rank, ex in enumerate(group_examples, start=1):
            t_idx = ex["t_idx"]
            dt = pd.Timestamp(times[t_idx])
            input_path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

            x_raw = torch.load(input_path, weights_only=True).float()
            x_norm = (x_raw - input_mean_cpu.view(1, 1, -1)) / input_std_cpu.view(1, 1, -1)

            example_prefix = f"{rank:02d}_{group_name}_tidx_{t_idx}"

            saliency_info = save_feature_saliency(
                model=get_model(metadata),  # fresh model instance because analysis.py loads weights into it
                weights_path=args.weights_path,
                input_tensor_norm=x_norm,
                feature_names=variable_names,
                output_dir=group_dir,
                example_prefix=example_prefix,
                device=device,
                target_index=args.apcp_index,
            )

            entry = {
                **ex,
                **saliency_info,
            }
            summary[group_name].append(entry)

            log(
                f"{group_name} rank {rank}: "
                f"t_idx={t_idx}, correct={ex['correct']}, "
                f"confidence={ex['confidence']:.4f}, "
                f"apcp_pred={ex['pred_apcp_value']:.4f}, "
                f"apcp_true={ex['true_apcp_value']:.4f}, "
                f"full_loss={ex['full_loss']:.4f}, "
                f"apcp_loss={ex['apcp_loss']:.4f}"
            )

    summary_path = args.output_dir / "confidence_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()