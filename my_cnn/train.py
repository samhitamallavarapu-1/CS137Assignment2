# CODEX OUTPUT
# import argparse
# import math
# import random
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# from model import get_model


# def choose_indices(times, year):
#     years = times.astype("datetime64[Y]").astype(int) + 1970
#     idx = np.where(years == year)[0]
#     return idx[idx + 24 < len(times)]


# def load_valid_indices(dataset_dir, times, index_list):
#     valid = []
#     for t_idx in tqdm(index_list, desc="Filtering NaN/missing inputs"):
#         dt = pd.Timestamp(times[t_idx])
#         path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
#         if not path.exists():
#             continue
#         try:
#             x = torch.load(path, weights_only=True).float()
#         except Exception:
#             continue
#         if torch.isnan(x).any():
#             continue
#         valid.append(int(t_idx))
#     return np.array(valid, dtype=int)


# def evaluate(model, dataset_dir, times, target_values, binary_labels, indices, device, input_mean, input_std):
#     model.eval()
#     all_preds = []
#     all_tgts = []
#     all_binary = []

#     with torch.no_grad():
#         for t_idx in indices:
#             dt = pd.Timestamp(times[t_idx])
#             path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
#             x = torch.load(path, weights_only=True).float().to(device)
#             x = (x - input_mean.to(device)) / input_std.to(device)
#             pred = model(x.unsqueeze(0)).squeeze(0).cpu()
#             tgt = target_values[t_idx + 24].cpu()
#             all_preds.append(pred)
#             all_tgts.append(tgt)
#             all_binary.append(binary_labels[t_idx + 24].float())

#     if len(all_preds) == 0:
#         raise RuntimeError("No samples found for evaluation")

#     preds = torch.stack(all_preds)
#     tgts = torch.stack(all_tgts)
#     binary = torch.stack(all_binary)

#     mse = torch.mean((preds - tgts) ** 2, dim=0)
#     rmse = torch.sqrt(mse)

#     apcp_idx = 5
#     rain_mask = tgts[:, apcp_idx] > 2.0
#     if rain_mask.sum() > 0:
#         rmse_apcp_rain = torch.sqrt(torch.mean((preds[rain_mask, apcp_idx] - tgts[rain_mask, apcp_idx]) ** 2)).item()
#     else:
#         rmse_apcp_rain = float('nan')

#     apcp_scores = (preds[:, apcp_idx] > 2.0).float().numpy()
#     apcp_labels = binary.numpy().astype(int)

#     try:
#         from sklearn.metrics import roc_auc_score

#         auc = float(roc_auc_score(apcp_labels, apcp_scores))
#     except Exception:
#         auc = float('nan')

#     metrics = {
#         'rmse': rmse.tolist(),
#         'rmse_apcp_rain': rmse_apcp_rain,
#         'auc_apcp': auc,
#         'n_samples': len(preds),
#     }
#     return metrics


# def main():
#     parser = argparse.ArgumentParser(description="Train weather CNN")
#     parser.add_argument("--dataset-root", type=Path, default=Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"))
#     parser.add_argument("--output-model", type=Path, default=Path("./my_cnn_weights.pth"))
#     parser.add_argument("--epochs", type=int, default=6)
#     parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (increased for speed)")
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--skip-test", action="store_true", help="Skip test evaluation at the end")
#     parser.add_argument("--early-stopping-patience", type=int, default=3, help="Patience for early stopping (epochs without improvement)")
#     parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
#     parser.add_argument("--rescan-nan", action="store_true", help="Force rescan for NaN values in inputs and recompute valid indices, ignoring cached files")
#     parser.add_argument("--skip-normalization", action="store_true", help="Skip input normalization and use identity (mean=0, std=1)")
#     args = parser.parse_args()

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     # load metadata + targets
#     dataset_dir = args.dataset_root
#     metadata_path = dataset_dir / "metadata.pt"
#     targets_path = dataset_dir / "targets.pt"

#     print(f"Loading metadata from {metadata_path}")
#     metadata = torch.load(metadata_path, weights_only=False)
#     print(f"Loading targets from {targets_path}")
#     targets_data = torch.load(targets_path, weights_only=False)

#     times = targets_data["time"]
#     target_values = targets_data["values"]
#     binary_labels = targets_data["binary_label"]

#     print("Preparing train/val/test folds using 2018-2024 data only …")
#     all_years = list(range(2018, 2025))  # 2018 to 2024

#     candidate_all = []
#     for year in all_years:
#         candidate_all.extend(choose_indices(times, year))
#     candidate_all = np.array(candidate_all, dtype=int)

#     print(f"Found {len(candidate_all)} candidate samples from 2018-2024")

#     cache_train_val_path = Path("valid_indices_train_val.npy")
#     cache_test_path = Path("valid_indices_test.npy")

#     if cache_train_val_path.exists() and cache_test_path.exists() and not args.rescan_nan:
#         print("Loading valid indices from cache...")
#         valid_train_val = np.load(cache_train_val_path)
#         valid_test = np.load(cache_test_path)
#     else:
#         if args.rescan_nan:
#             print("Forcing rescan for NaN values and recomputing valid indices...")
#         else:
#             print("No cache found; filtering NaN or missing inputs (this may take a few minutes)")
#         valid_all = load_valid_indices(dataset_dir, times, candidate_all)
        
#         if len(valid_all) < 2:
#             raise RuntimeError("Not enough valid samples after filtering")
        
#         # Split: 80% train/val, 20% test
#         rng = np.random.default_rng(args.seed)
#         shuffled_all = rng.permutation(valid_all)
#         split_idx = int(len(shuffled_all) * 0.8)
#         valid_train_val = shuffled_all[:split_idx]
#         valid_test = shuffled_all[split_idx:]
        
#         # Cache for future runs
#         np.save(cache_train_val_path, valid_train_val)
#         np.save(cache_test_path, valid_test)
#         print(f"Cached valid indices to {cache_train_val_path} and {cache_test_path}")
    
#     print(f"Valid samples: train_val={len(valid_train_val)}, test={len(valid_test)}")
#     if len(valid_train_val) < 2 or len(valid_test) < 1:
#         raise RuntimeError("Not enough valid samples after filtering for train/val or test")

#     # Compute input normalization stats from a sample of valid training data
#     print("Computing input normalization stats from sample of valid training data …")
#     sample_size = min(1000, len(valid_train_val))
#     sample_indices = np.random.choice(valid_train_val, size=sample_size, replace=False)
#     sample_inputs = []
#     for t_idx in sample_indices:
#         dt = pd.Timestamp(times[t_idx])
#         path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
#         try:
#             x = torch.load(path, weights_only=True).float()
#             if not torch.isnan(x).any():
#                 sample_inputs.append(x)
#         except:
#             pass
#     if len(sample_inputs) > 0:
#         sample_stack = torch.stack(sample_inputs)  # (N, 450, 449, c)
#         input_mean = sample_stack.mean(dim=[0,1,2])  # (c,)
#         input_std = sample_stack.std(dim=[0,1,2])   # (c,)
#         input_std = torch.clamp(input_std, min=1e-6)  # avoid div by zero
#         print(f"Input mean: {input_mean.tolist()[:5]}... (showing first 5)")
#         print(f"Input std: {input_std.tolist()[:5]}... (showing first 5)")
#     else:
#         input_mean = torch.zeros(metadata["n_vars"])
#         input_std = torch.ones(metadata["n_vars"])
#         print("Warning: No valid samples for normalization, using identity.")

#     # Shuffle train/val and split 80/20
#     rng = np.random.default_rng(args.seed)
#     shuffled_train_val = rng.permutation(valid_train_val)
#     n_train_val = len(shuffled_train_val)
#     n_train = int(n_train_val * 0.8)
#     n_val = n_train_val - n_train

#     train_idxs = shuffled_train_val[:n_train]
#     val_idxs = shuffled_train_val[n_train:]

#     test_idxs = valid_test  # no shuffle for test, keep temporal order

#     print(f"Total valid train/val samples after filtering: {n_train_val}")
#     print(f"Train samples: {len(train_idxs)}")
#     print(f"Val   samples: {len(val_idxs)}")
#     print(f"Test  samples: {len(test_idxs)}")

#     # Sanity check: confirm years
#     train_val_years_present = np.unique(times[valid_train_val].astype("datetime64[Y]").astype(int) + 1970)
#     test_years_present = np.unique(times[valid_test].astype("datetime64[Y]").astype(int) + 1970)
#     print(f"Train/val years represented: {train_val_years_present.tolist()}")
#     print(f"Test years represented: {test_years_present.tolist()}")

#     # Compute normalization stats from a subset of training data
#     if args.skip_normalization:
#         print("Skipping input normalization as requested.")
#         n_channels = metadata["n_vars"]
#         input_mean = torch.zeros(n_channels)
#         input_std = torch.ones(n_channels)
#     else:
#         print("Computing input normalization stats from training subset...")
#         subset_size = min(50, len(valid_train_val))
#         subset_indices = np.random.choice(valid_train_val, subset_size, replace=False)
#         n_channels = metadata["n_vars"]
        
#         # Incremental processing to minimize memory usage
#         sum_per_channel = torch.zeros(n_channels)
#         sum_sq_per_channel = torch.zeros(n_channels)
#         total_pixels = 0
        
#         for t_idx in tqdm(subset_indices, desc="Computing normalization stats"):
#             dt = pd.Timestamp(times[t_idx])
#             path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
#             x = torch.load(path, weights_only=True).float()
            
#             # x: (450, 449, c)
#             pixels_in_sample = x.shape[0] * x.shape[1]  # 450 * 449
#             total_pixels += pixels_in_sample
            
#             # Sum over height, width for each channel
#             sample_sum = x.sum(dim=[0,1])  # (c,)
#             sample_sum_sq = (x ** 2).sum(dim=[0,1])  # (c,)
            
#             sum_per_channel += sample_sum
#             sum_sq_per_channel += sample_sum_sq
            
#             # Explicitly delete to free memory
#             del x, sample_sum, sample_sum_sq
        
#         if total_pixels > 0:
#             input_mean = sum_per_channel / total_pixels
#             input_var = (sum_sq_per_channel / total_pixels) - (input_mean ** 2)
#             input_std = torch.sqrt(torch.clamp(input_var, min=0))
#             input_std = torch.clamp(input_std, min=1e-6)  # avoid div by zero
#             print(f"Input mean: {input_mean.tolist()[:5]}... (showing first 5)")
#             print(f"Input std: {input_std.tolist()[:5]}... (showing first 5)")
#         else:
#             input_mean = torch.zeros(n_channels)
#             input_std = torch.ones(n_channels)
#             print("Warning: No valid samples for normalization, using identity.")

#     # Save normalization stats
#     norm_path = Path("normalization_stats.pt")
#     torch.save({"input_mean": input_mean, "input_std": input_std}, norm_path)
#     print(f"Saved normalization stats to {norm_path}")


#     device = torch.device(args.device)
#     model = get_model(metadata).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     criterion = nn.MSELoss()

#     best_val_rmse = float('inf')
#     metrics_history = []
#     patience_counter = 0
#     early_stopping_enabled = not args.no_early_stopping

#     if early_stopping_enabled:
#         print(f"Early stopping enabled with patience={args.early_stopping_patience}")
#     else:
#         print("Early stopping disabled")

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         random.shuffle(train_idxs)
#         running_loss = 0.0

#         for i in range(0, len(train_idxs), args.batch_size):
#             batch_idxs = train_idxs[i:i + args.batch_size]
#             x_batch = []
#             y_batch = []

#             for t_idx in batch_idxs:
#                 dt = pd.Timestamp(times[t_idx])
#                 path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
#                 x = torch.load(path, weights_only=True).float()
#                 # Apply normalization
#                 x = (x - input_mean.view(1, 1, -1)) / input_std.view(1, 1, -1)
#                 y = target_values[t_idx + 24]
#                 x_batch.append(x)
#                 y_batch.append(y)

#             if len(x_batch) == 0:
#                 continue

#             x_batch = torch.stack(x_batch).to(device)
#             y_batch = torch.stack(y_batch).to(device)

#             optimizer.zero_grad()
#             pred = model(x_batch)
#             loss = criterion(pred, y_batch)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             running_loss += loss.item() * x_batch.shape[0]

#         epoch_loss = running_loss / max(1, len(train_idxs))
#         print(f"Epoch {epoch}/{args.epochs}: train loss = {epoch_loss:.6f}")

#         val_metrics = evaluate(model, dataset_dir, times, target_values, binary_labels, val_idxs, device, input_mean, input_std)
#         print(f"  Val RMSE (each var): {val_metrics['rmse']}")
#         print(f"  Val APCP>2mm RMSE: {val_metrics['rmse_apcp_rain']:.4f}, AUC: {val_metrics['auc_apcp']:.4f}")

#         metrics_history.append({"epoch": epoch, **val_metrics})

#         val_loss = float(np.mean(val_metrics['rmse']))
#         if val_loss < best_val_rmse:
#             best_val_rmse = val_loss
#             torch.save(model.state_dict(), args.output_model)
#             print(f"  Saved best model weights to {args.output_model}")
#             patience_counter = 0  # Reset patience
#         else:
#             patience_counter += 1
#             if early_stopping_enabled and patience_counter >= args.early_stopping_patience:
#                 print(f"Early stopping triggered: no improvement for {args.early_stopping_patience} epochs")
#                 break

#     print("Training complete.")

#     # Ensure a model file exists for evaluation (fallback to current model state)
#     if not args.output_model.exists():
#         torch.save(model.state_dict(), args.output_model)
#         print(f"No best model file found, saved final model weights to {args.output_model}")
#     else:
#         print(f"Best validation RMSE: {best_val_rmse:.6f}")

#     if not args.skip_test:
#         print("Evaluating test set...")
#         if args.output_model.exists():
#             model.load_state_dict(torch.load(args.output_model, map_location=device))
#         else:
#             print(f"WARNING: {args.output_model} not found. Evaluating current model state without loading.")

#         test_metrics = evaluate(model, dataset_dir, times, target_values, binary_labels, test_idxs, device, input_mean, input_std)
#         print(f"Test RMSE (each var): {test_metrics['rmse']}")
#         print(f"Test APCP>2mm RMSE: {test_metrics['rmse_apcp_rain']:.4f}, AUC: {test_metrics['auc_apcp']:.4f}")

#         metrics_history.append({"epoch": "test", **test_metrics})
#     else:
#         print("Skipping test evaluation as requested.")

#     history_path = Path("training_metrics.json")
#     import json
#     history_path.write_text(json.dumps(metrics_history, indent=2))
#     print(f"Metrics history saved to {history_path}")


# if __name__ == "__main__":
#     main()

#CHATGPT OUTPUT 1

# import argparse
# import json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm

# from model import get_model


# def choose_indices(times, year):
#     years = times.astype("datetime64[Y]").astype(int) + 1970
#     idx = np.where(years == year)[0]
#     return idx[idx + 24 < len(times)]


# def load_valid_indices(dataset_dir, times, index_list):
#     valid = []
#     for t_idx in tqdm(index_list, desc="Filtering NaN/missing inputs"):
#         dt = pd.Timestamp(times[t_idx])
#         path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

#         if not path.exists():
#             continue

#         try:
#             x = torch.load(path, weights_only=True).float()
#         except Exception:
#             continue

#         if torch.isnan(x).any():
#             continue

#         valid.append(int(t_idx))

#     return np.array(valid, dtype=int)


# def evaluate(model, dataset_dir, times, target_values, binary_labels, indices, device, input_mean, input_std):
#     model.eval()
#     all_preds = []
#     all_tgts = []
#     all_binary = []

#     input_mean = input_mean.to(device)
#     input_std = input_std.to(device)

#     with torch.no_grad():
#         for t_idx in indices:
#             dt = pd.Timestamp(times[t_idx])
#             path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

#             x = torch.load(path, weights_only=True).float().to(device)
#             x = (x - input_mean.view(1, 1, -1)) / input_std.view(1, 1, -1)

#             pred = model(x.unsqueeze(0)).squeeze(0).cpu()
#             tgt = target_values[t_idx + 24].cpu()

#             all_preds.append(pred)
#             all_tgts.append(tgt)
#             all_binary.append(binary_labels[t_idx + 24].float().cpu())

#     if len(all_preds) == 0:
#         raise RuntimeError("No samples found for evaluation")

#     preds = torch.stack(all_preds)
#     tgts = torch.stack(all_tgts)
#     binary = torch.stack(all_binary)

#     mse = torch.mean((preds - tgts) ** 2, dim=0)
#     rmse = torch.sqrt(mse)

#     apcp_idx = 5
#     rain_mask = tgts[:, apcp_idx] > 2.0
#     if rain_mask.sum() > 0:
#         rmse_apcp_rain = torch.sqrt(
#             torch.mean((preds[rain_mask, apcp_idx] - tgts[rain_mask, apcp_idx]) ** 2)
#         ).item()
#     else:
#         rmse_apcp_rain = float("nan")

#     apcp_scores = (preds[:, apcp_idx] > 2.0).float().numpy()
#     apcp_labels = binary.numpy().astype(int)

#     try:
#         from sklearn.metrics import roc_auc_score
#         auc = float(roc_auc_score(apcp_labels, apcp_scores))
#     except Exception:
#         auc = float("nan")

#     metrics = {
#         "rmse": rmse.tolist(),
#         "rmse_apcp_rain": rmse_apcp_rain,
#         "auc_apcp": auc,
#         "n_samples": len(preds),
#     }
#     return metrics


# def compute_normalization_stats(dataset_dir, times, indices, n_channels, max_samples=50):
#     if len(indices) == 0:
#         print("Warning: No training indices available for normalization. Using identity.")
#         return torch.zeros(n_channels), torch.ones(n_channels)

#     subset_size = min(max_samples, len(indices))
#     subset_indices = np.random.choice(indices, subset_size, replace=False)

#     sum_per_channel = torch.zeros(n_channels)
#     sum_sq_per_channel = torch.zeros(n_channels)
#     total_pixels = 0

#     for t_idx in tqdm(subset_indices, desc="Computing normalization stats"):
#         dt = pd.Timestamp(times[t_idx])
#         path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

#         x = torch.load(path, weights_only=True).float()

#         pixels_in_sample = x.shape[0] * x.shape[1]
#         total_pixels += pixels_in_sample

#         sum_per_channel += x.sum(dim=[0, 1])
#         sum_sq_per_channel += (x ** 2).sum(dim=[0, 1])

#         del x

#     if total_pixels == 0:
#         print("Warning: total_pixels == 0 while computing normalization. Using identity.")
#         return torch.zeros(n_channels), torch.ones(n_channels)

#     input_mean = sum_per_channel / total_pixels
#     input_var = (sum_sq_per_channel / total_pixels) - (input_mean ** 2)
#     input_std = torch.sqrt(torch.clamp(input_var, min=0.0))
#     input_std = torch.clamp(input_std, min=1e-6)

#     return input_mean, input_std


# def main():
#     parser = argparse.ArgumentParser(description="Train weather CNN")
#     parser.add_argument(
#         "--dataset-root",
#         type=Path,
#         default=Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"),
#     )
#     parser.add_argument("--output-model", type=Path, default=Path("./my_cnn_weights.pth"))
#     parser.add_argument("--epochs", type=int, default=6)
#     parser.add_argument("--batch-size", type=int, default=64)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda" if torch.cuda.is_available() else "cpu",
#     )
#     parser.add_argument("--skip-test", action="store_true", help="Skip test evaluation")
#     parser.add_argument(
#         "--early-stopping-patience",
#         type=int,
#         default=3,
#         help="Patience for early stopping",
#     )
#     parser.add_argument(
#         "--no-early-stopping",
#         action="store_true",
#         help="Disable early stopping",
#     )
#     parser.add_argument(
#         "--rescan-nan",
#         action="store_true",
#         help="Force rescan for NaN values and missing inputs",
#     )
#     parser.add_argument(
#         "--skip-normalization",
#         action="store_true",
#         help="Skip input normalization and use mean=0, std=1",
#     )
#     args = parser.parse_args()

#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     dataset_dir = args.dataset_root
#     metadata_path = dataset_dir / "metadata.pt"
#     targets_path = dataset_dir / "targets.pt"

#     print(f"Loading metadata from {metadata_path}")
#     metadata = torch.load(metadata_path, weights_only=False)

#     print(f"Loading targets from {targets_path}")
#     targets_data = torch.load(targets_path, weights_only=False)

#     times = targets_data["time"]
#     target_values = targets_data["values"]
#     binary_labels = targets_data["binary_label"]

#     print("Preparing train/val/test folds using 2018-2024 data only...")
#     all_years = list(range(2018, 2025))
#     candidate_all = []

#     for year in all_years:
#         candidate_all.extend(choose_indices(times, year))

#     candidate_all = np.array(candidate_all, dtype=int)
#     print(f"Found {len(candidate_all)} candidate samples from 2018-2024")

#     cache_train_val_path = Path("valid_indices_train_val.npy")
#     cache_test_path = Path("valid_indices_test.npy")

#     if cache_train_val_path.exists() and cache_test_path.exists() and not args.rescan_nan:
#         print("Loading valid indices from cache...")
#         valid_train_val = np.load(cache_train_val_path)
#         valid_test = np.load(cache_test_path)
#     else:
#         if args.rescan_nan:
#             print("Forcing rescan for NaN values and recomputing valid indices...")
#         else:
#             print("No cache found; filtering NaN or missing inputs (this may take a few minutes)...")

#         valid_all = load_valid_indices(dataset_dir, times, candidate_all)

#         if len(valid_all) < 2:
#             raise RuntimeError("Not enough valid samples after filtering")

#         rng = np.random.default_rng(args.seed)
#         shuffled_all = rng.permutation(valid_all)

#         split_idx = int(len(shuffled_all) * 0.8)
#         valid_train_val = shuffled_all[:split_idx]
#         valid_test = shuffled_all[split_idx:]

#         np.save(cache_train_val_path, valid_train_val)
#         np.save(cache_test_path, valid_test)
#         print(f"Cached valid indices to {cache_train_val_path} and {cache_test_path}")

#     print(f"Valid samples: train_val={len(valid_train_val)}, test={len(valid_test)}")
#     if len(valid_train_val) < 2 or len(valid_test) < 1:
#         raise RuntimeError("Not enough valid samples after filtering for train/val or test")

#     rng = np.random.default_rng(args.seed)
#     shuffled_train_val = rng.permutation(valid_train_val)

#     n_train_val = len(shuffled_train_val)
#     n_train = int(n_train_val * 0.8)

#     train_idxs = shuffled_train_val[:n_train]
#     val_idxs = shuffled_train_val[n_train:]
#     test_idxs = valid_test

#     print(f"Total valid train/val samples after filtering: {n_train_val}")
#     print(f"Train samples: {len(train_idxs)}")
#     print(f"Val   samples: {len(val_idxs)}")
#     print(f"Test  samples: {len(test_idxs)}")

#     train_val_years_present = np.unique(times[valid_train_val].astype("datetime64[Y]").astype(int) + 1970)
#     test_years_present = np.unique(times[valid_test].astype("datetime64[Y]").astype(int) + 1970)
#     print(f"Train/val years represented: {train_val_years_present.tolist()}")
#     print(f"Test years represented: {test_years_present.tolist()}")

#     n_channels = int(metadata["n_vars"])

#     if args.skip_normalization:
#         print("Skipping input normalization as requested.")
#         input_mean = torch.zeros(n_channels)
#         input_std = torch.ones(n_channels)
#     else:
#         print("Computing input normalization stats from training subset...")
#         input_mean, input_std = compute_normalization_stats(
#             dataset_dir=dataset_dir,
#             times=times,
#             indices=train_idxs,
#             n_channels=n_channels,
#             max_samples=50,
#         )
#         print(f"Input mean: {input_mean.tolist()[:5]}... (showing first 5)")
#         print(f"Input std: {input_std.tolist()[:5]}... (showing first 5)")

#     norm_path = Path("normalization_stats.pt")
#     torch.save({"input_mean": input_mean, "input_std": input_std}, norm_path)
#     print(f"Saved normalization stats to {norm_path}")

#     device = torch.device(args.device)
#     print(f"Using device: {device}")

#     model = get_model(metadata).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     criterion = nn.MSELoss()

#     best_val_rmse = float("inf")
#     metrics_history = []
#     patience_counter = 0
#     early_stopping_enabled = not args.no_early_stopping

#     if early_stopping_enabled:
#         print(f"Early stopping enabled with patience={args.early_stopping_patience}")
#     else:
#         print("Early stopping disabled")

#     input_mean_cpu = input_mean.cpu()
#     input_std_cpu = input_std.cpu()

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         np.random.shuffle(train_idxs)
#         running_loss = 0.0

#         for i in range(0, len(train_idxs), args.batch_size):
#             batch_idxs = train_idxs[i:i + args.batch_size]
#             x_batch = []
#             y_batch = []

#             for t_idx in batch_idxs:
#                 dt = pd.Timestamp(times[t_idx])
#                 path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

#                 x = torch.load(path, weights_only=True).float()
#                 x = (x - input_mean_cpu.view(1, 1, -1)) / input_std_cpu.view(1, 1, -1)

#                 y = target_values[t_idx + 24]
#                 x_batch.append(x)
#                 y_batch.append(y)

#             if len(x_batch) == 0:
#                 continue

#             x_batch = torch.stack(x_batch).to(device)
#             y_batch = torch.stack(y_batch).to(device)

#             optimizer.zero_grad()
#             pred = model(x_batch)
#             loss = criterion(pred, y_batch)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             running_loss += loss.item() * x_batch.shape[0]

#         epoch_loss = running_loss / max(1, len(train_idxs))
#         print(f"Epoch {epoch}/{args.epochs}: train loss = {epoch_loss:.6f}")

#         val_metrics = evaluate(
#             model=model,
#             dataset_dir=dataset_dir,
#             times=times,
#             target_values=target_values,
#             binary_labels=binary_labels,
#             indices=val_idxs,
#             device=device,
#             input_mean=input_mean_cpu,
#             input_std=input_std_cpu,
#         )

#         print(f"  Val RMSE (each var): {val_metrics['rmse']}")
#         print(
#             f"  Val APCP>2mm RMSE: {val_metrics['rmse_apcp_rain']:.4f}, "
#             f"AUC: {val_metrics['auc_apcp']:.4f}"
#         )

#         metrics_history.append({"epoch": epoch, **val_metrics})

#         val_loss = float(np.mean(val_metrics["rmse"]))
#         if val_loss < best_val_rmse:
#             best_val_rmse = val_loss
#             torch.save(model.state_dict(), args.output_model)
#             print(f"  Saved best model weights to {args.output_model}")
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if early_stopping_enabled and patience_counter >= args.early_stopping_patience:
#                 print(
#                     f"Early stopping triggered: no improvement for "
#                     f"{args.early_stopping_patience} epochs"
#                 )
#                 break

#     print("Training complete.")

#     if not args.output_model.exists():
#         torch.save(model.state_dict(), args.output_model)
#         print(f"No best model file found, saved final model weights to {args.output_model}")
#     else:
#         print(f"Best validation RMSE: {best_val_rmse:.6f}")

#     if not args.skip_test:
#         print("Evaluating test set...")
#         if args.output_model.exists():
#             model.load_state_dict(torch.load(args.output_model, map_location=device))
#         else:
#             print(
#                 f"WARNING: {args.output_model} not found. "
#                 "Evaluating current model state without loading."
#             )

#         test_metrics = evaluate(
#             model=model,
#             dataset_dir=dataset_dir,
#             times=times,
#             target_values=target_values,
#             binary_labels=binary_labels,
#             indices=test_idxs,
#             device=device,
#             input_mean=input_mean_cpu,
#             input_std=input_std_cpu,
#         )

#         print(f"Test RMSE (each var): {test_metrics['rmse']}")
#         print(
#             f"Test APCP>2mm RMSE: {test_metrics['rmse_apcp_rain']:.4f}, "
#             f"AUC: {test_metrics['auc_apcp']:.4f}"
#         )

#         metrics_history.append({"epoch": "test", **test_metrics})
#     else:
#         print("Skipping test evaluation as requested.")

#     history_path = Path("training_metrics.json")
#     history_path.write_text(json.dumps(metrics_history, indent=2))
#     print(f"Metrics history saved to {history_path}")


# if __name__ == "__main__":
#     main()

#CHATGPT OUTPUT 2

# import argparse
# import json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm

# from model import get_model


# def choose_indices(times, year):
#     years = times.astype("datetime64[Y]").astype(int) + 1970
#     idx = np.where(years == year)[0]
#     return idx[idx + 24 < len(times)]


# def load_valid_indices(dataset_dir, times, index_list, target_values, binary_labels):
#     valid = []

#     for t_idx in tqdm(index_list, desc="Filtering invalid/missing samples"):
#         if t_idx + 24 >= len(times):
#             continue

#         dt = pd.Timestamp(times[t_idx])
#         path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

#         if not path.exists():
#             continue

#         try:
#             x = torch.load(path, weights_only=True).float()
#         except Exception:
#             continue

#         y = target_values[t_idx + 24]
#         b = binary_labels[t_idx + 24].float()

#         if not torch.isfinite(x).all():
#             continue
#         if not torch.isfinite(y).all():
#             continue
#         if not torch.isfinite(b).all():
#             continue

#         valid.append(int(t_idx))

#     return np.array(valid, dtype=int)


# def evaluate(model, dataset_dir, times, target_values, binary_labels, indices, device, input_mean, input_std):
#     model.eval()
#     all_preds = []
#     all_tgts = []
#     all_binary = []

#     input_mean = input_mean.to(device)
#     input_std = input_std.to(device)

#     with torch.no_grad():
#         for t_idx in tqdm(indices, desc="Evaluating", leave=False):
#             if t_idx + 24 >= len(times):
#                 continue

#             dt = pd.Timestamp(times[t_idx])
#             path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

#             try:
#                 x = torch.load(path, weights_only=True).float()
#             except Exception:
#                 continue

#             tgt = target_values[t_idx + 24].cpu()
#             binary = binary_labels[t_idx + 24].float().cpu()

#             if not torch.isfinite(x).all():
#                 continue
#             if not torch.isfinite(tgt).all():
#                 continue
#             if not torch.isfinite(binary).all():
#                 continue

#             x = x.to(device)
#             x = (x - input_mean.view(1, 1, -1)) / input_std.view(1, 1, -1)

#             if not torch.isfinite(x).all():
#                 continue

#             pred = model(x.unsqueeze(0)).squeeze(0).cpu()

#             if not torch.isfinite(pred).all():
#                 continue

#             all_preds.append(pred)
#             all_tgts.append(tgt)
#             all_binary.append(binary)

#     if len(all_preds) == 0:
#         raise RuntimeError("No valid samples found for evaluation")

#     preds = torch.stack(all_preds)
#     tgts = torch.stack(all_tgts)
#     binary = torch.stack(all_binary)

#     mse = torch.mean((preds - tgts) ** 2, dim=0)
#     rmse = torch.sqrt(mse)

#     apcp_idx = 5
#     rain_mask = tgts[:, apcp_idx] > 2.0
#     if rain_mask.sum() > 0:
#         rmse_apcp_rain = torch.sqrt(
#             torch.mean((preds[rain_mask, apcp_idx] - tgts[rain_mask, apcp_idx]) ** 2)
#         ).item()
#     else:
#         rmse_apcp_rain = float("nan")

#     apcp_scores = preds[:, apcp_idx].numpy()
#     apcp_labels = binary.numpy().astype(int)

#     try:
#         from sklearn.metrics import roc_auc_score
#         auc = float(roc_auc_score(apcp_labels, apcp_scores))
#     except Exception:
#         auc = float("nan")

#     metrics = {
#         "rmse": rmse.tolist(),
#         "rmse_apcp_rain": rmse_apcp_rain,
#         "auc_apcp": auc,
#         "n_samples": len(preds),
#     }
#     return metrics


# def compute_normalization_stats(dataset_dir, times, indices, n_channels, max_samples=50):
#     if len(indices) == 0:
#         print("Warning: No training indices available for normalization. Using identity.")
#         return torch.zeros(n_channels), torch.ones(n_channels)

#     subset_size = min(max_samples, len(indices))
#     subset_indices = np.random.choice(indices, subset_size, replace=False)

#     sum_per_channel = torch.zeros(n_channels)
#     sum_sq_per_channel = torch.zeros(n_channels)
#     total_pixels = 0
#     n_used = 0

#     for t_idx in tqdm(subset_indices, desc="Computing normalization stats"):
#         if t_idx + 24 >= len(times):
#             continue

#         dt = pd.Timestamp(times[t_idx])
#         path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

#         try:
#             x = torch.load(path, weights_only=True).float()
#         except Exception:
#             continue

#         if not torch.isfinite(x).all():
#             continue

#         pixels_in_sample = x.shape[0] * x.shape[1]
#         total_pixels += pixels_in_sample
#         n_used += 1

#         sum_per_channel += x.sum(dim=[0, 1])
#         sum_sq_per_channel += (x ** 2).sum(dim=[0, 1])

#         del x

#     if total_pixels == 0 or n_used == 0:
#         print("Warning: No valid samples usable for normalization. Using identity.")
#         return torch.zeros(n_channels), torch.ones(n_channels)

#     input_mean = sum_per_channel / total_pixels
#     input_var = (sum_sq_per_channel / total_pixels) - (input_mean ** 2)
#     input_std = torch.sqrt(torch.clamp(input_var, min=0.0))
#     input_std = torch.clamp(input_std, min=1e-6)

#     return input_mean, input_std


# def main():
#     parser = argparse.ArgumentParser(description="Train weather CNN")
#     parser.add_argument(
#         "--dataset-root",
#         type=Path,
#         default=Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"),
#     )
#     parser.add_argument("--output-model", type=Path, default=Path("./my_cnn_weights.pth"))
#     parser.add_argument("--epochs", type=int, default=6)
#     parser.add_argument("--batch-size", type=int, default=64)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda" if torch.cuda.is_available() else "cpu",
#     )
#     parser.add_argument("--skip-test", action="store_true", help="Skip test evaluation")
#     parser.add_argument(
#         "--early-stopping-patience",
#         type=int,
#         default=3,
#         help="Patience for early stopping",
#     )
#     parser.add_argument(
#         "--no-early-stopping",
#         action="store_true",
#         help="Disable early stopping",
#     )
#     parser.add_argument(
#         "--rescan-nan",
#         action="store_true",
#         help="Force rescan for NaN/inf values and missing inputs",
#     )
#     parser.add_argument(
#         "--skip-normalization",
#         action="store_true",
#         help="Skip input normalization and use mean=0, std=1",
#     )
#     args = parser.parse_args()

#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     dataset_dir = args.dataset_root
#     metadata_path = dataset_dir / "metadata.pt"
#     targets_path = dataset_dir / "targets.pt"

#     print(f"Loading metadata from {metadata_path}")
#     metadata = torch.load(metadata_path, weights_only=False)

#     print(f"Loading targets from {targets_path}")
#     targets_data = torch.load(targets_path, weights_only=False)

#     times = targets_data["time"]
#     target_values = targets_data["values"]
#     binary_labels = targets_data["binary_label"]

#     print("All target_values finite:", torch.isfinite(target_values).all().item())
#     print("All binary_labels finite:", torch.isfinite(binary_labels.float()).all().item())

#     print("Preparing train/val/test folds using 2018-2024 data only...")
#     all_years = list(range(2018, 2025))
#     candidate_all = []

#     for year in all_years:
#         candidate_all.extend(choose_indices(times, year))

#     candidate_all = np.array(candidate_all, dtype=int)
#     print(f"Found {len(candidate_all)} candidate samples from 2018-2024")

#     cache_train_val_path = Path("valid_indices_train_val.npy")
#     cache_test_path = Path("valid_indices_test.npy")

#     if cache_train_val_path.exists() and cache_test_path.exists() and not args.rescan_nan:
#         print("Loading valid indices from cache...")
#         valid_train_val = np.load(cache_train_val_path)
#         valid_test = np.load(cache_test_path)
#     else:
#         if args.rescan_nan:
#             print("Forcing rescan for NaN/inf values and recomputing valid indices...")
#         else:
#             print("No cache found; filtering invalid or missing inputs (this may take a few minutes)...")

#         valid_all = load_valid_indices(dataset_dir, times, candidate_all, target_values, binary_labels)

#         if len(valid_all) < 2:
#             raise RuntimeError("Not enough valid samples after filtering")

#         rng = np.random.default_rng(args.seed)
#         shuffled_all = rng.permutation(valid_all)

#         split_idx = int(len(shuffled_all) * 0.8)
#         valid_train_val = shuffled_all[:split_idx]
#         valid_test = shuffled_all[split_idx:]

#         np.save(cache_train_val_path, valid_train_val)
#         np.save(cache_test_path, valid_test)
#         print(f"Cached valid indices to {cache_train_val_path} and {cache_test_path}")

#     print(f"Valid samples: train_val={len(valid_train_val)}, test={len(valid_test)}")
#     if len(valid_train_val) < 2 or len(valid_test) < 1:
#         raise RuntimeError("Not enough valid samples after filtering for train/val or test")

#     rng = np.random.default_rng(args.seed)
#     shuffled_train_val = rng.permutation(valid_train_val)

#     n_train_val = len(shuffled_train_val)
#     n_train = int(n_train_val * 0.8)

#     train_idxs = shuffled_train_val[:n_train]
#     val_idxs = shuffled_train_val[n_train:]
#     test_idxs = valid_test

#     print(f"Total valid train/val samples after filtering: {n_train_val}")
#     print(f"Train samples: {len(train_idxs)}")
#     print(f"Val   samples: {len(val_idxs)}")
#     print(f"Test  samples: {len(test_idxs)}")

#     train_val_years_present = np.unique(times[valid_train_val].astype("datetime64[Y]").astype(int) + 1970)
#     test_years_present = np.unique(times[valid_test].astype("datetime64[Y]").astype(int) + 1970)
#     print(f"Train/val years represented: {train_val_years_present.tolist()}")
#     print(f"Test years represented: {test_years_present.tolist()}")

#     n_channels = int(metadata["n_vars"])

#     if args.skip_normalization:
#         print("Skipping input normalization as requested.")
#         input_mean = torch.zeros(n_channels)
#         input_std = torch.ones(n_channels)
#     else:
#         print("Computing input normalization stats from training subset...")
#         input_mean, input_std = compute_normalization_stats(
#             dataset_dir=dataset_dir,
#             times=times,
#             indices=train_idxs,
#             n_channels=n_channels,
#             max_samples=50,
#         )
#         print(f"Input mean: {input_mean.tolist()[:5]}... (showing first 5)")
#         print(f"Input std: {input_std.tolist()[:5]}... (showing first 5)")

#     norm_path = Path("normalization_stats.pt")
#     torch.save({"input_mean": input_mean, "input_std": input_std}, norm_path)
#     print(f"Saved normalization stats to {norm_path}")

#     device = torch.device(args.device)
#     print(f"Using device: {device}")

#     model = get_model(metadata).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     criterion = nn.MSELoss()

#     best_val_rmse = float("inf")
#     metrics_history = []
#     patience_counter = 0
#     early_stopping_enabled = not args.no_early_stopping

#     if early_stopping_enabled:
#         print(f"Early stopping enabled with patience={args.early_stopping_patience}")
#     else:
#         print("Early stopping disabled")

#     input_mean_cpu = input_mean.cpu()
#     input_std_cpu = input_std.cpu()

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         np.random.shuffle(train_idxs)
#         running_loss = 0.0
#         n_train_used = 0

#         num_batches = (len(train_idxs) + args.batch_size - 1) // args.batch_size
#         pbar = tqdm(
#             range(0, len(train_idxs), args.batch_size),
#             total=num_batches,
#             desc=f"Epoch {epoch}",
#             leave=False,
#         )

#         for i in pbar:
#             batch_idxs = train_idxs[i:i + args.batch_size]
#             x_batch = []
#             y_batch = []

#             for t_idx in batch_idxs:
#                 if t_idx + 24 >= len(times):
#                     continue

#                 dt = pd.Timestamp(times[t_idx])
#                 path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

#                 try:
#                     x = torch.load(path, weights_only=True).float()
#                 except Exception:
#                     continue

#                 y = target_values[t_idx + 24]

#                 if not torch.isfinite(x).all():
#                     continue
#                 if not torch.isfinite(y).all():
#                     continue

#                 x = (x - input_mean_cpu.view(1, 1, -1)) / input_std_cpu.view(1, 1, -1)

#                 if not torch.isfinite(x).all():
#                     continue

#                 x_batch.append(x)
#                 y_batch.append(y)

#             if len(x_batch) == 0:
#                 pbar.set_postfix({"loss": "skip"})
#                 continue

#             x_batch = torch.stack(x_batch).to(device)
#             y_batch = torch.stack(y_batch).to(device)

#             optimizer.zero_grad()
#             pred = model(x_batch)

#             if not torch.isfinite(pred).all():
#                 print(f"[TRAIN WARNING] Non-finite prediction in epoch {epoch}, batch starting at {i}; skipping batch")
#                 continue

#             loss = criterion(pred, y_batch)

#             if not torch.isfinite(loss):
#                 print(f"[TRAIN WARNING] Non-finite loss in epoch {epoch}, batch starting at {i}; skipping batch")
#                 continue

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             running_loss += loss.item() * x_batch.shape[0]
#             n_train_used += x_batch.shape[0]

#             pbar.set_postfix({"loss": f"{loss.item():.4f}"})

#         epoch_loss = running_loss / max(1, n_train_used)
#         print(f"Epoch {epoch}/{args.epochs}: train loss = {epoch_loss:.6f} (used {n_train_used} samples)")

#         val_metrics = evaluate(
#             model=model,
#             dataset_dir=dataset_dir,
#             times=times,
#             target_values=target_values,
#             binary_labels=binary_labels,
#             indices=val_idxs,
#             device=device,
#             input_mean=input_mean_cpu,
#             input_std=input_std_cpu,
#         )

#         print(f"  Val RMSE (each var): {val_metrics['rmse']}")
#         print(
#             f"  Val APCP>2mm RMSE: {val_metrics['rmse_apcp_rain']:.4f}, "
#             f"AUC: {val_metrics['auc_apcp']:.4f}"
#         )

#         metrics_history.append({"epoch": epoch, **val_metrics})

#         val_loss = float(np.mean(val_metrics["rmse"]))
#         if np.isfinite(val_loss) and val_loss < best_val_rmse:
#             best_val_rmse = val_loss
#             torch.save(model.state_dict(), args.output_model)
#             print(f"  Saved best model weights to {args.output_model}")
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if early_stopping_enabled and patience_counter >= args.early_stopping_patience:
#                 print(
#                     f"Early stopping triggered: no improvement for "
#                     f"{args.early_stopping_patience} epochs"
#                 )
#                 break

#     print("Training complete.")

#     if not args.output_model.exists():
#         torch.save(model.state_dict(), args.output_model)
#         print(f"No best model file found, saved final model weights to {args.output_model}")
#     else:
#         if np.isfinite(best_val_rmse):
#             print(f"Best validation RMSE: {best_val_rmse:.6f}")
#         else:
#             print("Best validation RMSE was not finite.")

#     if not args.skip_test:
#         print("Evaluating test set...")
#         if args.output_model.exists():
#             model.load_state_dict(torch.load(args.output_model, map_location=device))
#         else:
#             print(
#                 f"WARNING: {args.output_model} not found. "
#                 "Evaluating current model state without loading."
#             )

#         test_metrics = evaluate(
#             model=model,
#             dataset_dir=dataset_dir,
#             times=times,
#             target_values=target_values,
#             binary_labels=binary_labels,
#             indices=test_idxs,
#             device=device,
#             input_mean=input_mean_cpu,
#             input_std=input_std_cpu,
#         )

#         print(f"Test RMSE (each var): {test_metrics['rmse']}")
#         print(
#             f"Test APCP>2mm RMSE: {test_metrics['rmse_apcp_rain']:.4f}, "
#             f"AUC: {test_metrics['auc_apcp']:.4f}"
#         )

#         metrics_history.append({"epoch": "test", **test_metrics})
#     else:
#         print("Skipping test evaluation as requested.")

#     history_path = Path("training_metrics.json")
#     history_path.write_text(json.dumps(metrics_history, indent=2))
#     print(f"Metrics history saved to {history_path}")


# if __name__ == "__main__":
#     main()

# CHATGPT OUTPUT 3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import get_model


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


def evaluate(model, dataset_dir, times, target_values, binary_labels, indices, device, input_mean, input_std):
    model.eval()
    all_preds = []
    all_tgts = []
    all_binary = []

    input_mean = input_mean.to(device)
    input_std = input_std.to(device)

    with torch.no_grad():
        for t_idx in tqdm(indices, desc="Evaluating", leave=False):
            if t_idx + 24 >= len(times):
                continue

            dt = pd.Timestamp(times[t_idx])
            path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

            try:
                x = torch.load(path, weights_only=True).float()
            except Exception:
                continue

            tgt = target_values[t_idx + 24].cpu()
            binary = binary_labels[t_idx + 24].float().cpu()

            if not torch.isfinite(x).all():
                continue
            if not torch.isfinite(tgt).all():
                continue
            if not torch.isfinite(binary).all():
                continue

            x = x.to(device)
            x = (x - input_mean.view(1, 1, -1)) / input_std.view(1, 1, -1)

            if not torch.isfinite(x).all():
                continue

            pred = model(x.unsqueeze(0)).squeeze(0).cpu()

            if not torch.isfinite(pred).all():
                continue

            all_preds.append(pred)
            all_tgts.append(tgt)
            all_binary.append(binary)

    if len(all_preds) == 0:
        raise RuntimeError("No valid samples found for evaluation")

    preds = torch.stack(all_preds)
    tgts = torch.stack(all_tgts)
    binary = torch.stack(all_binary)

    mse = torch.mean((preds - tgts) ** 2, dim=0)
    rmse = torch.sqrt(mse)

    apcp_idx = 5
    rain_mask = tgts[:, apcp_idx] > 2.0
    if rain_mask.sum() > 0:
        rmse_apcp_rain = torch.sqrt(
            torch.mean((preds[rain_mask, apcp_idx] - tgts[rain_mask, apcp_idx]) ** 2)
        ).item()
    else:
        rmse_apcp_rain = float("nan")

    apcp_scores = preds[:, apcp_idx].numpy()
    apcp_labels = binary.numpy().astype(int)

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(apcp_labels, apcp_scores))
    except Exception:
        auc = float("nan")

    metrics = {
        "rmse": rmse.tolist(),
        "rmse_apcp_rain": rmse_apcp_rain,
        "auc_apcp": auc,
        "n_samples": len(preds),
    }
    return metrics


def compute_normalization_stats(dataset_dir, times, indices, n_channels, batch_size=64):
    """
    Compute per-channel normalization stats across the entire provided index set
    using a running sum / sum-of-squares accumulation over batches.
    """
    if len(indices) == 0:
        print("Warning: No indices available for normalization. Using identity.")
        return torch.zeros(n_channels), torch.ones(n_channels)

    sum_per_channel = torch.zeros(n_channels, dtype=torch.float64)
    sum_sq_per_channel = torch.zeros(n_channels, dtype=torch.float64)
    total_pixels = 0
    n_used = 0

    num_batches = (len(indices) + batch_size - 1) // batch_size
    pbar = tqdm(
        range(0, len(indices), batch_size),
        total=num_batches,
        desc="Computing normalization stats",
        leave=False,
    )

    for start in pbar:
        batch_indices = indices[start:start + batch_size]

        for t_idx in batch_indices:
            if t_idx + 24 >= len(times):
                continue

            dt = pd.Timestamp(times[t_idx])
            path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

            try:
                x = torch.load(path, weights_only=True).float()
            except Exception:
                continue

            if not torch.isfinite(x).all():
                continue

            pixels_in_sample = x.shape[0] * x.shape[1]
            total_pixels += pixels_in_sample
            n_used += 1

            sum_per_channel += x.sum(dim=[0, 1], dtype=torch.float64)
            sum_sq_per_channel += (x ** 2).sum(dim=[0, 1], dtype=torch.float64)

            del x

        pbar.set_postfix({"used": n_used})

    if total_pixels == 0 or n_used == 0:
        print("Warning: No valid samples usable for normalization. Using identity.")
        return torch.zeros(n_channels), torch.ones(n_channels)

    input_mean = sum_per_channel / total_pixels
    input_var = (sum_sq_per_channel / total_pixels) - (input_mean ** 2)
    input_std = torch.sqrt(torch.clamp(input_var, min=0.0))
    input_std = torch.clamp(input_std, min=1e-6)

    return input_mean.float(), input_std.float()


def main():
    parser = argparse.ArgumentParser(description="Train weather CNN")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"),
    )
    parser.add_argument("--output-model", type=Path, default=Path("./my_cnn_weights.pth"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip test evaluation")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--rescan-nan",
        action="store_true",
        help="Force rescan for NaN/inf values and missing inputs",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Load normalization stats from normalization_stats.pt instead of recomputing",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    print("All target_values finite:", torch.isfinite(target_values).all().item())
    print("All binary_labels finite:", torch.isfinite(binary_labels.float()).all().item())

    print("Preparing train/val/test folds using 2018-2024 data only...")
    all_years = list(range(2018, 2025))
    candidate_all = []

    for year in all_years:
        candidate_all.extend(choose_indices(times, year))

    candidate_all = np.array(candidate_all, dtype=int)
    print(f"Found {len(candidate_all)} candidate samples from 2018-2024")

    cache_train_val_path = Path("valid_indices_train_val.npy")
    cache_test_path = Path("valid_indices_test.npy")

    if cache_train_val_path.exists() and cache_test_path.exists() and not args.rescan_nan:
        print("Loading valid indices from cache...")
        valid_train_val = np.load(cache_train_val_path)
        valid_test = np.load(cache_test_path)
    else:
        if args.rescan_nan:
            print("Forcing rescan for NaN/inf values and recomputing valid indices...")
        else:
            print("No cache found; filtering invalid or missing inputs (this may take a few minutes)...")

        valid_all = load_valid_indices(dataset_dir, times, candidate_all, target_values, binary_labels)

        if len(valid_all) < 2:
            raise RuntimeError("Not enough valid samples after filtering")

        rng = np.random.default_rng(args.seed)
        shuffled_all = rng.permutation(valid_all)

        split_idx = int(len(shuffled_all) * 0.8)
        valid_train_val = shuffled_all[:split_idx]
        valid_test = shuffled_all[split_idx:]

        np.save(cache_train_val_path, valid_train_val)
        np.save(cache_test_path, valid_test)
        print(f"Cached valid indices to {cache_train_val_path} and {cache_test_path}")

    print(f"Valid samples: train_val={len(valid_train_val)}, test={len(valid_test)}")
    if len(valid_train_val) < 2 or len(valid_test) < 1:
        raise RuntimeError("Not enough valid samples after filtering for train/val or test")

    rng = np.random.default_rng(args.seed)
    shuffled_train_val = rng.permutation(valid_train_val)

    n_train_val = len(shuffled_train_val)
    n_train = int(n_train_val * 0.8)

    train_idxs = shuffled_train_val[:n_train]
    val_idxs = shuffled_train_val[n_train:]
    test_idxs = valid_test

    print(f"Total valid train/val samples after filtering: {n_train_val}")
    print(f"Train samples: {len(train_idxs)}")
    print(f"Val   samples: {len(val_idxs)}")
    print(f"Test  samples: {len(test_idxs)}")

    train_val_years_present = np.unique(times[valid_train_val].astype("datetime64[Y]").astype(int) + 1970)
    test_years_present = np.unique(times[valid_test].astype("datetime64[Y]").astype(int) + 1970)
    print(f"Train/val years represented: {train_val_years_present.tolist()}")
    print(f"Test years represented: {test_years_present.tolist()}")

    n_channels = int(metadata["n_vars"])
    norm_path = Path("normalization_stats.pt")

    if args.skip_normalization:
        if norm_path.exists():
            print(f"Loading saved normalization stats from {norm_path}")
            norm_stats = torch.load(norm_path, weights_only=False)
            input_mean = norm_stats["input_mean"].float()
            input_std = norm_stats["input_std"].float()
            input_std = torch.clamp(input_std, min=1e-6)
            print(f"Loaded input mean: {input_mean.tolist()[:5]}... (showing first 5)")
            print(f"Loaded input std: {input_std.tolist()[:5]}... (showing first 5)")
        else:
            print(f"Warning: {norm_path} not found. Falling back to identity normalization.")
            input_mean = torch.zeros(n_channels)
            input_std = torch.ones(n_channels)
    else:
        print("Computing input normalization stats from the entire development set (2018-2024 valid train/val samples)...")
        input_mean, input_std = compute_normalization_stats(
            dataset_dir=dataset_dir,
            times=times,
            indices=valid_train_val,
            n_channels=n_channels,
            batch_size=args.batch_size,
        )
        print(f"Input mean: {input_mean.tolist()[:5]}... (showing first 5)")
        print(f"Input std: {input_std.tolist()[:5]}... (showing first 5)")

        torch.save({"input_mean": input_mean, "input_std": input_std}, norm_path)
        print(f"Saved normalization stats to {norm_path}")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = get_model(metadata).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    metrics_history = []
    patience_counter = 0
    early_stopping_enabled = not args.no_early_stopping

    if early_stopping_enabled:
        print(f"Early stopping enabled with patience={args.early_stopping_patience}")
    else:
        print("Early stopping disabled")

    input_mean_cpu = input_mean.cpu()
    input_std_cpu = input_std.cpu()

    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(train_idxs)
        running_loss = 0.0
        n_train_used = 0

        num_batches = (len(train_idxs) + args.batch_size - 1) // args.batch_size
        pbar = tqdm(
            range(0, len(train_idxs), args.batch_size),
            total=num_batches,
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for i in pbar:
            batch_idxs = train_idxs[i:i + args.batch_size]
            x_batch = []
            y_batch = []

            for t_idx in batch_idxs:
                if t_idx + 24 >= len(times):
                    continue

                dt = pd.Timestamp(times[t_idx])
                path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

                try:
                    x = torch.load(path, weights_only=True).float()
                except Exception:
                    continue

                y = target_values[t_idx + 24]

                if not torch.isfinite(x).all():
                    continue
                if not torch.isfinite(y).all():
                    continue

                x = (x - input_mean_cpu.view(1, 1, -1)) / input_std_cpu.view(1, 1, -1)

                if not torch.isfinite(x).all():
                    continue

                x_batch.append(x)
                y_batch.append(y)

            if len(x_batch) == 0:
                pbar.set_postfix({"loss": "skip"})
                continue

            x_batch = torch.stack(x_batch).to(device)
            y_batch = torch.stack(y_batch).to(device)

            optimizer.zero_grad()
            pred = model(x_batch)

            if not torch.isfinite(pred).all():
                print(f"[TRAIN WARNING] Non-finite prediction in epoch {epoch}, batch starting at {i}; skipping batch")
                continue

            loss = criterion(pred, y_batch)

            if not torch.isfinite(loss):
                print(f"[TRAIN WARNING] Non-finite loss in epoch {epoch}, batch starting at {i}; skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x_batch.shape[0]
            n_train_used += x_batch.shape[0]

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / max(1, n_train_used)
        print(f"Epoch {epoch}/{args.epochs}: train loss = {epoch_loss:.6f} (used {n_train_used} samples)")

        val_metrics = evaluate(
            model=model,
            dataset_dir=dataset_dir,
            times=times,
            target_values=target_values,
            binary_labels=binary_labels,
            indices=val_idxs,
            device=device,
            input_mean=input_mean_cpu,
            input_std=input_std_cpu,
        )

        print(f"  Val RMSE (each var): {val_metrics['rmse']}")
        print(
            f"  Val APCP>2mm RMSE: {val_metrics['rmse_apcp_rain']:.4f}, "
            f"AUC: {val_metrics['auc_apcp']:.4f}"
        )

        metrics_history.append({"epoch": epoch, **val_metrics})

        val_loss = float(np.mean(val_metrics["rmse"]))
        if np.isfinite(val_loss) and val_loss < best_val_rmse:
            best_val_rmse = val_loss
            torch.save(model.state_dict(), args.output_model)
            print(f"  Saved best model weights to {args.output_model}")
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stopping_enabled and patience_counter >= args.early_stopping_patience:
                print(
                    f"Early stopping triggered: no improvement for "
                    f"{args.early_stopping_patience} epochs"
                )
                break

    print("Training complete.")

    if not args.output_model.exists():
        torch.save(model.state_dict(), args.output_model)
        print(f"No best model file found, saved final model weights to {args.output_model}")
    else:
        if np.isfinite(best_val_rmse):
            print(f"Best validation RMSE: {best_val_rmse:.6f}")
        else:
            print("Best validation RMSE was not finite.")

    if not args.skip_test:
        print("Evaluating test set...")
        if args.output_model.exists():
            model.load_state_dict(torch.load(args.output_model, map_location=device))
        else:
            print(
                f"WARNING: {args.output_model} not found. "
                "Evaluating current model state without loading."
            )

        test_metrics = evaluate(
            model=model,
            dataset_dir=dataset_dir,
            times=times,
            target_values=target_values,
            binary_labels=binary_labels,
            indices=test_idxs,
            device=device,
            input_mean=input_mean_cpu,
            input_std=input_std_cpu,
        )

        print(f"Test RMSE (each var): {test_metrics['rmse']}")
        print(
            f"Test APCP>2mm RMSE: {test_metrics['rmse_apcp_rain']:.4f}, "
            f"AUC: {test_metrics['auc_apcp']:.4f}"
        )

        metrics_history.append({"epoch": "test", **test_metrics})
    else:
        print("Skipping test evaluation as requested.")

    history_path = Path("training_metrics.json")
    history_path.write_text(json.dumps(metrics_history, indent=2))
    print(f"Metrics history saved to {history_path}")


if __name__ == "__main__":
    main()