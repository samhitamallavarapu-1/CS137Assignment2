import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from mycnn2_model import build_model


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_indices(times, years):
    years_array = times.astype("datetime64[Y]").astype(int) + 1970
    mask = np.isin(years_array, years)
    idx = np.where(mask)[0]
    return idx[idx + 24 < len(times)]


def load_valid_indices(dataset_dir, times, index_list, target_values, binary_labels):
    valid = []

    for t_idx in tqdm(index_list, desc="Filtering valid samples"):
        if t_idx + 24 >= len(times):
            continue

        dt = pd.Timestamp(times[t_idx])
        x_path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

        if not x_path.exists():
            continue

        try:
            x = torch.load(x_path, weights_only=True).float()
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


def compute_normalization_stats(dataset_dir, times, indices, n_channels):
    sum_per_channel = torch.zeros(n_channels, dtype=torch.float64)
    sumsq_per_channel = torch.zeros(n_channels, dtype=torch.float64)
    total_pixels = 0

    for t_idx in tqdm(indices, desc="Computing normalization stats"):
        dt = pd.Timestamp(times[t_idx])
        x_path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

        x = torch.load(x_path, weights_only=True).float()

        pixels = x.shape[0] * x.shape[1]
        total_pixels += pixels

        sum_per_channel += x.sum(dim=(0, 1), dtype=torch.float64)
        sumsq_per_channel += (x ** 2).sum(dim=(0, 1), dtype=torch.float64)

    mean = sum_per_channel / total_pixels
    var = (sumsq_per_channel / total_pixels) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    std = torch.clamp(std, min=1e-6)

    return mean.float(), std.float()


def load_batch(dataset_dir, times, target_values, indices, input_mean, input_std, device):
    x_batch = []
    y_batch = []

    for t_idx in indices:
        dt = pd.Timestamp(times[t_idx])
        x_path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

        x = torch.load(x_path, weights_only=True).float()
        y = target_values[t_idx + 24].float()

        x = (x - input_mean.view(1, 1, -1)) / input_std.view(1, 1, -1)

        x_batch.append(x)
        y_batch.append(y)

    x_batch = torch.stack(x_batch).to(device)   # (B, H, W, C)
    y_batch = torch.stack(y_batch).to(device)   # (B, 6)
    return x_batch, y_batch


def evaluate(model, dataset_dir, times, target_values, binary_labels, indices, batch_size, input_mean, input_std, device):
    model.eval()

    preds_all = []
    tgts_all = []
    binary_all = []

    with torch.no_grad():
        for start in tqdm(range(0, len(indices), batch_size), desc="Evaluating", leave=False):
            batch_idxs = indices[start:start + batch_size]
            x_batch, y_batch = load_batch(
                dataset_dir, times, target_values, batch_idxs, input_mean, input_std, device
            )

            preds = model(x_batch).cpu()
            preds_all.append(preds)
            tgts_all.append(y_batch.cpu())

            for t_idx in batch_idxs:
                binary_all.append(binary_labels[t_idx + 24].item())

    preds = torch.cat(preds_all, dim=0)
    tgts = torch.cat(tgts_all, dim=0)
    binary = torch.tensor(binary_all).bool()

    rmse = torch.sqrt(torch.mean((preds - tgts) ** 2, dim=0))

    apcp_idx = 5
    rain_mask = tgts[:, apcp_idx] > 2.0
    if rain_mask.sum() > 0:
        rmse_apcp_rain = torch.sqrt(
            torch.mean((preds[rain_mask, apcp_idx] - tgts[rain_mask, apcp_idx]) ** 2)
        ).item()
    else:
        rmse_apcp_rain = float("nan")

    try:
        from sklearn.metrics import roc_auc_score
        auc_apcp = float(roc_auc_score(binary.numpy().astype(int), preds[:, apcp_idx].numpy()))
    except Exception:
        auc_apcp = float("nan")

    return {
        "rmse": rmse.tolist(),
        "rmse_mean": float(rmse.mean().item()),
        "rmse_apcp_rain": rmse_apcp_rain,
        "auc_apcp": auc_apcp,
        "n_samples": int(len(indices)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train baseline WeatherCNN")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-years", nargs="+", type=int, default=[2018, 2019, 2020, 2021])
    parser.add_argument("--val-years", nargs="+", type=int, default=[2022])
    parser.add_argument("--rescan-valid", action="store_true", help="Force re-filtering valid samples")
    parser.add_argument("--recompute-normalization", action="store_true", help="Force recomputing normalization stats")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_root
    metadata = torch.load(dataset_dir / "metadata.pt", weights_only=False)
    targets_data = torch.load(dataset_dir / "targets.pt", weights_only=False)

    times = targets_data["time"]
    target_values = targets_data["values"]
    binary_labels = targets_data["binary_label"]

    train_candidates = choose_indices(times, args.train_years)
    val_candidates = choose_indices(times, args.val_years)

    print(f"Train candidate samples: {len(train_candidates)}")
    print(f"Val candidate samples: {len(val_candidates)}")

    train_cache_path = args.output_dir / "train_indices.npy"
    val_cache_path = args.output_dir / "val_indices.npy"

    if train_cache_path.exists() and val_cache_path.exists() and not args.rescan_valid:
        train_indices = np.load(train_cache_path)
        val_indices = np.load(val_cache_path)
        print(f"Loaded cached train indices from {train_cache_path}")
        print(f"Loaded cached val indices from {val_cache_path}")
    else:
        train_indices = load_valid_indices(
            dataset_dir, times, train_candidates, target_values, binary_labels
        )
        val_indices = load_valid_indices(
            dataset_dir, times, val_candidates, target_values, binary_labels
        )

        np.save(train_cache_path, train_indices)
        np.save(val_cache_path, val_indices)
        print(f"Saved cached train indices to {train_cache_path}")
        print(f"Saved cached val indices to {val_cache_path}")

    print(f"Valid train samples: {len(train_indices)}")
    print(f"Valid val samples: {len(val_indices)}")

    n_channels = int(metadata["n_vars"])
    norm_path = args.output_dir / "normalization_stats.pt"

    if norm_path.exists() and not args.recompute_normalization:
        norm_stats = torch.load(norm_path, weights_only=False)
        input_mean = norm_stats["input_mean"].float()
        input_std = torch.clamp(norm_stats["input_std"].float(), min=1e-6)
        print(f"Loaded normalization stats from {norm_path}")
    else:
        input_mean, input_std = compute_normalization_stats(
            dataset_dir, times, train_indices, n_channels
        )
        torch.save(
            {"input_mean": input_mean, "input_std": input_std},
            norm_path,
        )
        print(f"Saved normalization stats to {norm_path}")

    device = torch.device(args.device)

    model = build_model(metadata).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    history = []

    best_model_path = args.output_dir / "my_cnn2_weights.pth"
    metrics_path = args.output_dir / "my_cnn2_training_metrics.json"

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = np.random.permutation(train_indices)

        running_loss = 0.0
        n_seen = 0

        for start in tqdm(range(0, len(shuffled), args.batch_size), desc=f"Epoch {epoch}/{args.epochs}"):
            batch_idxs = shuffled[start:start + args.batch_size]

            x_batch, y_batch = load_batch(
                dataset_dir, times, target_values, batch_idxs, input_mean, input_std, device
            )

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x_batch.shape[0]
            n_seen += x_batch.shape[0]

        train_loss = running_loss / max(1, n_seen)

        val_metrics = evaluate(
            model=model,
            dataset_dir=dataset_dir,
            times=times,
            target_values=target_values,
            binary_labels=binary_labels,
            indices=val_indices,
            batch_size=args.batch_size,
            input_mean=input_mean,
            input_std=input_std,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_rmse_mean": val_metrics["rmse_mean"],
            "val_rmse": val_metrics["rmse"],
            "val_rmse_apcp_rain": val_metrics["rmse_apcp_rain"],
            "val_auc_apcp": val_metrics["auc_apcp"],
            "n_train_samples": int(len(train_indices)),
            "n_val_samples": int(len(val_indices)),
        }
        history.append(row)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss:         {train_loss:.6f}")
        print(f"  Val RMSE mean:      {val_metrics['rmse_mean']:.6f}")
        print(f"  Val APCP rain RMSE: {val_metrics['rmse_apcp_rain']:.6f}")
        print(f"  Val APCP AUC:       {val_metrics['auc_apcp']:.6f}")
        print(f"  Val RMSE by var:    {val_metrics['rmse']}")

        if val_metrics["rmse_mean"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse_mean"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model to {best_model_path}")

        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

    print("\nTraining complete.")
    print(f"Best validation RMSE mean: {best_val_rmse:.6f}")


if __name__ == "__main__":
    main()