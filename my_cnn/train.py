import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from model import get_model


def choose_indices(times, year):
    years = times.astype("datetime64[Y]").astype(int) + 1970
    idx = np.where(years == year)[0]
    return idx[idx + 24 < len(times)]


def load_valid_indices(dataset_dir, times, index_list):
    valid = []
    for t_idx in index_list:
        dt = pd.Timestamp(times[t_idx])
        path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
        if not path.exists():
            continue
        try:
            x = torch.load(path, weights_only=True).float()
        except Exception:
            continue
        if torch.isnan(x).any():
            continue
        valid.append(int(t_idx))
    return np.array(valid, dtype=int)


def evaluate(model, dataset_dir, times, target_values, binary_labels, indices, device, input_mean, input_std):
    model.eval()
    all_preds = []
    all_tgts = []
    all_binary = []

    with torch.no_grad():
        for t_idx in indices:
            dt = pd.Timestamp(times[t_idx])
            path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
            x = torch.load(path, weights_only=True).float().to(device)
            x = (x - input_mean.to(device)) / input_std.to(device)
            pred = model(x.unsqueeze(0)).squeeze(0).cpu()
            tgt = target_values[t_idx + 24].cpu()
            all_preds.append(pred)
            all_tgts.append(tgt)
            all_binary.append(binary_labels[t_idx + 24].float())

    if len(all_preds) == 0:
        raise RuntimeError("No samples found for evaluation")

    preds = torch.stack(all_preds)
    tgts = torch.stack(all_tgts)
    binary = torch.stack(all_binary)

    mse = torch.mean((preds - tgts) ** 2, dim=0)
    rmse = torch.sqrt(mse)

    apcp_idx = 5
    rain_mask = tgts[:, apcp_idx] > 2.0
    if rain_mask.sum() > 0:
        rmse_apcp_rain = torch.sqrt(torch.mean((preds[rain_mask, apcp_idx] - tgts[rain_mask, apcp_idx]) ** 2)).item()
    else:
        rmse_apcp_rain = float('nan')

    apcp_scores = preds[:, apcp_idx].numpy()
    apcp_labels = binary.numpy().astype(int)

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(apcp_labels, apcp_scores))
    except Exception:
        auc = float('nan')

    metrics = {
        'rmse': rmse.tolist(),
        'rmse_apcp_rain': rmse_apcp_rain,
        'auc_apcp': auc,
        'n_samples': len(preds),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train weather CNN")
    parser.add_argument("--dataset-root", type=Path, default=Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"))
    parser.add_argument("--output-model", type=Path, default=Path("./my_cnn_weights.pth"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (increased for speed)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-test", action="store_true", help="Skip test evaluation at the end")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Patience for early stopping (epochs without improvement)")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load metadata + targets
    dataset_dir = args.dataset_root
    metadata_path = dataset_dir / "metadata.pt"
    targets_path = dataset_dir / "targets.pt"

    print(f"Loading metadata from {metadata_path}")
    metadata = torch.load(metadata_path, weights_only=False)
    print(f"Loading targets from {targets_path}")
    targets_data = torch.load(targets_path, weights_only=False)

    times = targets_data["time"]
    target_values = targets_data["values"]
    binary_labels = targets_data["binary_label"]

    print("Preparing train/val/test folds: train/val from 2018-2024, test from 2025 …")
    train_val_years = list(range(2018, 2025))  # 2018 to 2024
    test_years = [2025]

    candidate_train_val = []
    for year in train_val_years:
        candidate_train_val.extend(choose_indices(times, year))
    candidate_train_val = np.array(candidate_train_val, dtype=int)

    candidate_test = []
    for year in test_years:
        candidate_test.extend(choose_indices(times, year))
    candidate_test = np.array(candidate_test, dtype=int)

    # Check for cached valid indices
    cache_train_val_path = Path("valid_indices_train_val.npy")
    cache_test_path = Path("valid_indices_test.npy")

    if cache_train_val_path.exists() and cache_test_path.exists():
        print("Loading valid indices from cache...")
        valid_train_val = np.load(cache_train_val_path)
        valid_test = np.load(cache_test_path)
    else:
        print("Filtering NaN or missing inputs for train/val (this may take a few minutes)")
        valid_train_val = load_valid_indices(dataset_dir, times, candidate_train_val)
        print("Filtering NaN or missing inputs for test")
        valid_test = load_valid_indices(dataset_dir, times, candidate_test)
        # Cache for future runs
        np.save(cache_train_val_path, valid_train_val)
        np.save(cache_test_path, valid_test)
        print(f"Cached valid indices to {cache_train_val_path} and {cache_test_path}")

    if len(valid_train_val) < 2:
        raise RuntimeError("Not enough valid train/val samples after filtering")
    if len(valid_test) < 1:
        raise RuntimeError("Not enough valid test samples after filtering")

    # Compute input normalization stats from a sample of valid training data
    print("Computing input normalization stats from sample of valid training data …")
    sample_size = min(1000, len(valid_train_val))
    sample_indices = np.random.choice(valid_train_val, size=sample_size, replace=False)
    sample_inputs = []
    for t_idx in sample_indices:
        dt = pd.Timestamp(times[t_idx])
        path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
        try:
            x = torch.load(path, weights_only=True).float()
            if not torch.isnan(x).any():
                sample_inputs.append(x)
        except:
            pass
    if len(sample_inputs) > 0:
        sample_stack = torch.stack(sample_inputs)  # (N, 450, 449, c)
        input_mean = sample_stack.mean(dim=[0,1,2])  # (c,)
        input_std = sample_stack.std(dim=[0,1,2])   # (c,)
        input_std = torch.clamp(input_std, min=1e-6)  # avoid div by zero
        print(f"Input mean: {input_mean.tolist()[:5]}... (showing first 5)")
        print(f"Input std: {input_std.tolist()[:5]}... (showing first 5)")
    else:
        input_mean = torch.zeros(metadata["n_vars"])
        input_std = torch.ones(metadata["n_vars"])
        print("Warning: No valid samples for normalization, using identity.")

    # Shuffle train/val and split 80/20
    rng = np.random.default_rng(args.seed)
    shuffled_train_val = rng.permutation(valid_train_val)
    n_train_val = len(shuffled_train_val)
    n_train = int(n_train_val * 0.8)
    n_val = n_train_val - n_train

    train_idxs = shuffled_train_val[:n_train]
    val_idxs = shuffled_train_val[n_train:]

    test_idxs = valid_test  # no shuffle for test, keep temporal order

    print(f"Total valid train/val samples after filtering: {n_train_val}")
    print(f"Train samples: {len(train_idxs)}")
    print(f"Val   samples: {len(val_idxs)}")
    print(f"Test  samples: {len(test_idxs)}")

    # Sanity check: confirm years
    train_val_years_present = np.unique(times[valid_train_val].astype("datetime64[Y]").astype(int) + 1970)
    test_years_present = np.unique(times[valid_test].astype("datetime64[Y]").astype(int) + 1970)
    print(f"Train/val years represented: {train_val_years_present.tolist()}")
    print(f"Test years represented: {test_years_present.tolist()}")

    # Compute normalization stats from a subset of training data
    print("Computing input normalization stats from training subset...")
    subset_size = min(1000, len(valid_train_val))
    subset_indices = np.random.choice(valid_train_val, subset_size, replace=False)
    channel_means = []
    channel_stds = []
    n_channels = metadata["n_vars"]
    for _ in range(n_channels):
        channel_means.append([])
        channel_stds.append([])

    for t_idx in subset_indices:
        dt = pd.Timestamp(times[t_idx])
        path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
        x = torch.load(path, weights_only=True).float()
        if torch.isnan(x).any():
            continue
        # x: (450, 449, c)
        for ch in range(n_channels):
            ch_data = x[:, :, ch]
            channel_means[ch].append(ch_data.mean().item())
            channel_stds[ch].append(ch_data.std().item())

    input_mean = torch.tensor([np.mean(channel_means[ch]) for ch in range(n_channels)])
    input_std = torch.tensor([max(np.mean(channel_stds[ch]), 1e-6) for ch in range(n_channels)])  # avoid div by 0
    print(f"Input normalization: mean shape {input_mean.shape}, std shape {input_std.shape}")

    # Save normalization stats
    norm_path = Path("normalization_stats.pt")
    torch.save({"input_mean": input_mean, "input_std": input_std}, norm_path)
    print(f"Saved normalization stats to {norm_path}")


    device = torch.device(args.device)
    model = get_model(metadata).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    metrics_history = []
    patience_counter = 0
    early_stopping_enabled = not args.no_early_stopping

    if early_stopping_enabled:
        print(f"Early stopping enabled with patience={args.early_stopping_patience}")
    else:
        print("Early stopping disabled")

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_idxs)
        running_loss = 0.0

        for i in range(0, len(train_idxs), args.batch_size):
            batch_idxs = train_idxs[i:i + args.batch_size]
            x_batch = []
            y_batch = []

            for t_idx in batch_idxs:
                dt = pd.Timestamp(times[t_idx])
                path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
                x = torch.load(path, weights_only=True).float()
                # Apply normalization
                x = (x - input_mean.view(1, 1, -1)) / input_std.view(1, 1, -1)
                y = target_values[t_idx + 24]
                x_batch.append(x)
                y_batch.append(y)

            if len(x_batch) == 0:
                continue

            x_batch = torch.stack(x_batch).to(device)
            y_batch = torch.stack(y_batch).to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x_batch.shape[0]

        epoch_loss = running_loss / max(1, len(train_idxs))
        print(f"Epoch {epoch}/{args.epochs}: train loss = {epoch_loss:.6f}")

        val_metrics = evaluate(model, dataset_dir, times, target_values, binary_labels, val_idxs, device, input_mean, input_std)
        print(f"  Val RMSE (each var): {val_metrics['rmse']}")
        print(f"  Val APCP>2mm RMSE: {val_metrics['rmse_apcp_rain']:.4f}, AUC: {val_metrics['auc_apcp']:.4f}")

        metrics_history.append({"epoch": epoch, **val_metrics})

        val_loss = float(np.mean(val_metrics['rmse']))
        if val_loss < best_val_rmse:
            best_val_rmse = val_loss
            torch.save(model.state_dict(), args.output_model)
            print(f"  Saved best model weights to {args.output_model}")
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
            if early_stopping_enabled and patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered: no improvement for {args.early_stopping_patience} epochs")
                break

    print("Training complete.")

    # Ensure a model file exists for evaluation (fallback to current model state)
    if not args.output_model.exists():
        torch.save(model.state_dict(), args.output_model)
        print(f"No best model file found, saved final model weights to {args.output_model}")
    else:
        print(f"Best validation RMSE: {best_val_rmse:.6f}")

    if not args.skip_test:
        print("Evaluating test set...")
        if args.output_model.exists():
            model.load_state_dict(torch.load(args.output_model, map_location=device))
        else:
            print(f"WARNING: {args.output_model} not found. Evaluating current model state without loading.")

        test_metrics = evaluate(model, dataset_dir, times, target_values, binary_labels, test_idxs, device, input_mean, input_std)
        print(f"Test RMSE (each var): {test_metrics['rmse']}")
        print(f"Test APCP>2mm RMSE: {test_metrics['rmse_apcp_rain']:.4f}, AUC: {test_metrics['auc_apcp']:.4f}")

        metrics_history.append({"epoch": "test", **test_metrics})
    else:
        print("Skipping test evaluation as requested.")

    history_path = Path("training_metrics.json")
    import json
    history_path.write_text(json.dumps(metrics_history, indent=2))
    print(f"Metrics history saved to {history_path}")


if __name__ == "__main__":
    main()
