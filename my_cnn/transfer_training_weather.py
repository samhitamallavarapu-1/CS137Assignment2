import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# Logging
# ============================================================

def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ============================================================
# Dataset utilities
# ============================================================

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


class WeatherTransferDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        times,
        target_values: torch.Tensor,
        indices: np.ndarray,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
    ):
        self.dataset_dir = dataset_dir
        self.times = times
        self.target_values = target_values
        self.indices = np.array(indices, dtype=int)
        self.input_mean = input_mean.float().view(1, 1, -1)
        self.input_std = input_std.float().view(1, 1, -1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_idx = int(self.indices[idx])
        dt = pd.Timestamp(self.times[t_idx])
        path = self.dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

        x = torch.load(path, weights_only=True).float()
        y = self.target_values[t_idx + 24].float()

        x = (x - self.input_mean) / self.input_std
        # channels-last -> channels-first for torchvision models
        x = x.permute(2, 0, 1).contiguous()

        return x, y


# ============================================================
# Normalization
# ============================================================

def compute_normalization_stats(dataset_dir, times, indices, n_channels, batch_size=64):
    """
    Compute per-channel normalization stats across the full provided index set
    using running sums / sumsq.
    """
    if len(indices) == 0:
        log("Warning: No indices available for normalization. Using identity.")
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
        log("Warning: No valid samples usable for normalization. Using identity.")
        return torch.zeros(n_channels), torch.ones(n_channels)

    input_mean = sum_per_channel / total_pixels
    input_var = (sum_sq_per_channel / total_pixels) - (input_mean ** 2)
    input_std = torch.sqrt(torch.clamp(input_var, min=0.0))
    input_std = torch.clamp(input_std, min=1e-6)

    return input_mean.float(), input_std.float()


def get_or_create_normalization_stats(
    dataset_dir: Path,
    times,
    valid_train_val: np.ndarray,
    n_channels: int,
    stats_path: Path,
    recompute: bool,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if stats_path.exists() and not recompute:
        log(f"Loading cached normalization stats from {stats_path}")
        stats = torch.load(stats_path, weights_only=False)
        input_mean = stats["input_mean"].float()
        input_std = torch.clamp(stats["input_std"].float(), min=1e-6)
        return input_mean, input_std

    log("Computing normalization stats from full development set (valid train/val pool)...")
    input_mean, input_std = compute_normalization_stats(
        dataset_dir=dataset_dir,
        times=times,
        indices=valid_train_val,
        n_channels=n_channels,
        batch_size=batch_size,
    )

    torch.save({"input_mean": input_mean, "input_std": input_std}, stats_path)
    log(f"Saved normalization stats to {stats_path}")
    return input_mean, input_std


# ============================================================
# Model creation / weight loading
# ============================================================

def _get_default_weights(model_name: str):
    weights_lookup = {
        "resnet18": tv_models.ResNet18_Weights.DEFAULT,
        "resnet34": tv_models.ResNet34_Weights.DEFAULT,
        "resnet50": tv_models.ResNet50_Weights.DEFAULT,
        "resnet101": tv_models.ResNet101_Weights.DEFAULT,
        "resnet152": tv_models.ResNet152_Weights.DEFAULT,
        "densenet121": tv_models.DenseNet121_Weights.DEFAULT,
        "densenet169": tv_models.DenseNet169_Weights.DEFAULT,
        "densenet201": tv_models.DenseNet201_Weights.DEFAULT,
        "efficientnet_b0": tv_models.EfficientNet_B0_Weights.DEFAULT,
        "efficientnet_b1": tv_models.EfficientNet_B1_Weights.DEFAULT,
        "efficientnet_b2": tv_models.EfficientNet_B2_Weights.DEFAULT,
        "efficientnet_b3": tv_models.EfficientNet_B3_Weights.DEFAULT,
        "mobilenet_v3_small": tv_models.MobileNet_V3_Small_Weights.DEFAULT,
        "mobilenet_v3_large": tv_models.MobileNet_V3_Large_Weights.DEFAULT,
        "vit_b_16": tv_models.ViT_B_16_Weights.DEFAULT,
        "vit_b_32": tv_models.ViT_B_32_Weights.DEFAULT,
    }
    if model_name not in weights_lookup:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return weights_lookup[model_name]


def adapt_first_conv_layer(model: nn.Module, n_input_channels: int) -> nn.Module:
    """
    Adapt first convolution layer to the weather input channel count.
    For pretrained conv weights, copy/average intelligently.
    """
    # ResNet-style stem
    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        old_conv = model.conv1
        if old_conv.in_channels == n_input_channels:
            return model

        new_conv = nn.Conv2d(
            in_channels=n_input_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        with torch.no_grad():
            if old_conv.weight.shape[1] == 3:
                if n_input_channels == 1:
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                elif n_input_channels < 3:
                    new_conv.weight[:, :n_input_channels].copy_(old_conv.weight[:, :n_input_channels])
                else:
                    new_conv.weight[:, :3].copy_(old_conv.weight)
                    mean_channel = old_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(3, n_input_channels):
                        new_conv.weight[:, c:c+1].copy_(mean_channel)

            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        model.conv1 = new_conv
        return model

    # EfficientNet / MobileNet features[0][0] often first conv
    if hasattr(model, "features") and len(list(model.features.children())) > 0:
        first_block = list(model.features.children())[0]

        # Direct conv
        if isinstance(first_block, nn.Conv2d):
            old_conv = first_block
            if old_conv.in_channels == n_input_channels:
                return model

            new_conv = nn.Conv2d(
                in_channels=n_input_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                if old_conv.weight.shape[1] == 3:
                    new_conv.weight[:, :min(3, n_input_channels)].copy_(old_conv.weight[:, :min(3, n_input_channels)])
                    if n_input_channels > 3:
                        mean_channel = old_conv.weight.mean(dim=1, keepdim=True)
                        for c in range(3, n_input_channels):
                            new_conv.weight[:, c:c+1].copy_(mean_channel)
            model.features[0] = new_conv
            return model

        # ConvNormActivation-like sequential with first element conv
        if isinstance(first_block, nn.Sequential) and len(first_block) > 0 and isinstance(first_block[0], nn.Conv2d):
            old_conv = first_block[0]
            if old_conv.in_channels == n_input_channels:
                return model

            new_conv = nn.Conv2d(
                in_channels=n_input_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                if old_conv.weight.shape[1] == 3:
                    new_conv.weight[:, :min(3, n_input_channels)].copy_(old_conv.weight[:, :min(3, n_input_channels)])
                    if n_input_channels > 3:
                        mean_channel = old_conv.weight.mean(dim=1, keepdim=True)
                        for c in range(3, n_input_channels):
                            new_conv.weight[:, c:c+1].copy_(mean_channel)
            first_block[0] = new_conv
            model.features[0] = first_block
            return model

    raise ValueError("Could not adapt first conv layer for this architecture.")


def get_pretrained_backbone(
    model_name: str = "resnet50",
    n_outputs: int = 6,
    n_input_channels: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name in {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}:
        constructor = getattr(tv_models, model_name)
        model = constructor(weights=_get_default_weights(model_name) if pretrained else None)
        model = adapt_first_conv_layer(model, n_input_channels)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_outputs)
        return model

    if model_name in {"densenet121", "densenet169", "densenet201"}:
        constructor = getattr(tv_models, model_name)
        model = constructor(weights=_get_default_weights(model_name) if pretrained else None)
        model = adapt_first_conv_layer(model, n_input_channels)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, n_outputs)
        return model

    if model_name in {"efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"}:
        constructor = getattr(tv_models, model_name)
        model = constructor(weights=_get_default_weights(model_name) if pretrained else None)
        model = adapt_first_conv_layer(model, n_input_channels)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, n_outputs)
        return model

    if model_name in {"mobilenet_v3_small", "mobilenet_v3_large"}:
        constructor = getattr(tv_models, model_name)
        model = constructor(weights=_get_default_weights(model_name) if pretrained else None)
        model = adapt_first_conv_layer(model, n_input_channels)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, n_outputs)
        return model

    if model_name in {"vit_b_16", "vit_b_32"}:
        constructor = getattr(tv_models, model_name)
        model = constructor(weights=_get_default_weights(model_name) if pretrained else None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, n_outputs)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def load_model_weights(model: nn.Module, weights_path: str, device: Optional[torch.device] = None) -> nn.Module:
    map_location = device if device is not None else "cpu"
    state_dict = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


# ============================================================
# Freeze / unfreeze
# ============================================================

def get_final_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        return model.fc

    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            return model.classifier
        if isinstance(model.classifier, nn.Sequential):
            return model.classifier[-1]

    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        return model.heads.head

    raise ValueError("Could not identify final layer for this model.")


def get_trainable_blocks(model: nn.Module) -> List[nn.Module]:
    # ResNet-like
    if hasattr(model, "fc") and hasattr(model, "layer4"):
        return [
            model.fc,
            model.layer4,
            model.layer3,
            model.layer2,
            model.layer1,
            nn.Sequential(model.conv1, model.bn1),
        ]

    # DenseNet
    if hasattr(model, "features") and hasattr(model, "classifier") and hasattr(model.features, "denseblock4"):
        blocks = [
            get_final_layer(model),
            model.features.denseblock4,
            model.features.transition3,
            model.features.denseblock3,
            model.features.transition2,
            model.features.denseblock2,
            model.features.transition1,
            model.features.denseblock1,
        ]
        if hasattr(model.features, "conv0"):
            blocks.append(model.features.conv0)
        return blocks

    # EfficientNet / MobileNet
    if hasattr(model, "features") and hasattr(model, "classifier"):
        return [get_final_layer(model)] + list(model.features.children())[::-1]

    # ViT
    if hasattr(model, "encoder") and hasattr(model, "heads"):
        return [get_final_layer(model)] + list(model.encoder.layers.children())[::-1]

    return [get_final_layer(model)]


def freeze_all_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def set_train_mode_full(model: nn.Module) -> None:
    unfreeze_all_parameters(model)


def set_train_mode_last_layer(model: nn.Module) -> None:
    freeze_all_parameters(model)
    final_layer = get_final_layer(model)
    for p in final_layer.parameters():
        p.requires_grad = True


def set_train_mode_gradual_unfreeze(model: nn.Module, epoch: int, unfreeze_every: int = 1) -> int:
    freeze_all_parameters(model)
    blocks = get_trainable_blocks(model)

    n_blocks_unfrozen = min(
        len(blocks),
        1 + (max(epoch, 1) - 1) // max(unfreeze_every, 1),
    )

    for block in blocks[:n_blocks_unfrozen]:
        for p in block.parameters():
            p.requires_grad = True

    return n_blocks_unfrozen


def get_optimizer_for_trainable_params(model: nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found.")
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


# ============================================================
# Metrics
# ============================================================

def evaluate_model(model, dataloader, device, criterion) -> Dict:
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_preds = []
    all_tgts = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)

            if not torch.isfinite(preds).all():
                continue

            loss = criterion(preds, targets)
            if not torch.isfinite(loss):
                continue

            batch_size = inputs.shape[0]
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            all_preds.append(preds.detach().cpu())
            all_tgts.append(targets.detach().cpu())

    if n_samples == 0:
        raise RuntimeError("No valid evaluation samples found.")

    preds = torch.cat(all_preds, dim=0)
    tgts = torch.cat(all_tgts, dim=0)

    mse = torch.mean((preds - tgts) ** 2, dim=0)
    rmse = torch.sqrt(mse)

    metrics = {
        "loss": float(running_loss / n_samples),
        "rmse_mean": float(rmse.mean().item()),
        "rmse": rmse.tolist(),
        "n_samples": int(n_samples),
    }

    # APCP-specific metrics if output 5 exists
    if preds.shape[1] > 5:
        apcp_idx = 5
        rain_mask = tgts[:, apcp_idx] > 2.0
        if rain_mask.sum() > 0:
            rmse_apcp_rain = torch.sqrt(
                torch.mean((preds[rain_mask, apcp_idx] - tgts[rain_mask, apcp_idx]) ** 2)
            ).item()
        else:
            rmse_apcp_rain = float("nan")

        metrics["rmse_apcp_rain"] = rmse_apcp_rain

        try:
            from sklearn.metrics import roc_auc_score
            apcp_scores = preds[:, apcp_idx].numpy()
            apcp_labels = (tgts[:, apcp_idx] > 2.0).int().numpy()
            metrics["auc_apcp"] = float(roc_auc_score(apcp_labels, apcp_scores))
        except Exception:
            metrics["auc_apcp"] = float("nan")

    return metrics


# ============================================================
# Training
# ============================================================

def train_transfer_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    train_mode: str,
    unfreeze_every: int,
    best_model_path: Path,
    final_model_path: Path,
    history_path: Path,
    early_stopping_patience: Optional[int],
) -> List[Dict]:
    model = model.to(device)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    metrics_history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        if train_mode == "full":
            set_train_mode_full(model)
            strategy_note = "all parameters trainable"
        elif train_mode == "last_layer":
            set_train_mode_last_layer(model)
            strategy_note = "only final layer trainable"
        elif train_mode == "gradual_unfreeze":
            n_blocks = set_train_mode_gradual_unfreeze(
                model=model,
                epoch=epoch,
                unfreeze_every=unfreeze_every,
            )
            strategy_note = f"{n_blocks} output-side blocks trainable"
        else:
            raise ValueError(f"Unsupported train_mode: {train_mode}")

        optimizer = get_optimizer_for_trainable_params(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
        )

        model.train()
        running_loss = 0.0
        n_train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [{train_mode}]", leave=False)

        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(inputs)

            if not torch.isfinite(preds).all():
                pbar.set_postfix({"loss": "nonfinite_pred"})
                continue

            loss = criterion(preds, targets)

            if not torch.isfinite(loss):
                pbar.set_postfix({"loss": "nonfinite_loss"})
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = inputs.shape[0]
            running_loss += loss.item() * batch_size
            n_train_samples += batch_size

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / max(1, n_train_samples)

        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            criterion=criterion,
        )
        val_loss = val_metrics["loss"]

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_rmse_mean": float(val_metrics["rmse_mean"]),
            "val_rmse": val_metrics["rmse"],
            "strategy": train_mode,
            "strategy_note": strategy_note,
            "n_train_samples": int(n_train_samples),
            "n_val_samples": int(val_metrics["n_samples"]),
        }

        if "rmse_apcp_rain" in val_metrics:
            epoch_metrics["val_rmse_apcp_rain"] = val_metrics["rmse_apcp_rain"]
        if "auc_apcp" in val_metrics:
            epoch_metrics["val_auc_apcp"] = val_metrics["auc_apcp"]

        metrics_history.append(epoch_metrics)

        log(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_rmse_mean={val_metrics['rmse_mean']:.6f} | "
            f"{strategy_note}"
        )

        history_path.write_text(json.dumps(metrics_history, indent=2))
        log(f"Updated metrics history at {history_path}")

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            log(f"Saved best model weights to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            log(
                f"Early stopping triggered after {early_stopping_patience} "
                f"epochs without improvement."
            )
            break

    torch.save(model.state_dict(), final_model_path)
    log(f"Saved final model weights to {final_model_path}")

    return metrics_history


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Transfer learning weather predictor trainer")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        help="torchvision model name (default: resnet50)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision pretrained weights for backbone",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional path to a saved .pth state_dict to load after model creation",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="full",
        choices=["full", "last_layer", "gradual_unfreeze"],
    )
    parser.add_argument("--unfreeze-every", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--rescan-nan", action="store_true")
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path("normalization_stats_transfer.pt"),
        help="Path for cached normalization stats",
    )
    parser.add_argument(
        "--recompute-normalization",
        action="store_true",
        help="Force recomputation of normalization stats even if cache exists",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Optional early stopping patience",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for metrics JSON and model .pth outputs",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_root
    metadata_path = dataset_dir / "metadata.pt"
    targets_path = dataset_dir / "targets.pt"

    log(f"Loading metadata from {metadata_path}")
    metadata = torch.load(metadata_path, weights_only=False)

    log(f"Loading targets from {targets_path}")
    targets_data = torch.load(targets_path, weights_only=False)

    times = targets_data["time"]
    target_values = targets_data["values"]
    binary_labels = targets_data["binary_label"]

    log(f"All target_values finite: {torch.isfinite(target_values).all().item()}")
    log(f"All binary_labels finite: {torch.isfinite(binary_labels.float()).all().item()}")

    log("Preparing train/val/test folds using 2018-2024 data only...")
    all_years = list(range(2018, 2025))
    candidate_all = []

    for year in all_years:
        candidate_all.extend(choose_indices(times, year))

    candidate_all = np.array(candidate_all, dtype=int)
    log(f"Found {len(candidate_all)} candidate samples from 2018-2024")

    cache_train_val_path = args.output_dir / "valid_indices_train_val.npy"
    cache_test_path = args.output_dir / "valid_indices_test.npy"

    if cache_train_val_path.exists() and cache_test_path.exists() and not args.rescan_nan:
        log("Loading valid indices from cache...")
        valid_train_val = np.load(cache_train_val_path)
        valid_test = np.load(cache_test_path)
    else:
        if args.rescan_nan:
            log("Forcing rescan for NaN/inf values and recomputing valid indices...")
        else:
            log("No cache found; filtering invalid or missing inputs (this may take a few minutes)...")

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
        log(f"Cached valid indices to {cache_train_val_path} and {cache_test_path}")

    log(f"Valid samples: train_val={len(valid_train_val)}, test={len(valid_test)}")
    if len(valid_train_val) < 2 or len(valid_test) < 1:
        raise RuntimeError("Not enough valid samples after filtering for train/val or test")

    rng = np.random.default_rng(args.seed)
    shuffled_train_val = rng.permutation(valid_train_val)

    n_train_val = len(shuffled_train_val)
    n_train = int(n_train_val * 0.8)

    train_idxs = shuffled_train_val[:n_train]
    val_idxs = shuffled_train_val[n_train:]
    test_idxs = valid_test

    log(f"Total valid train/val samples after filtering: {n_train_val}")
    log(f"Train samples: {len(train_idxs)}")
    log(f"Val samples: {len(val_idxs)}")
    log(f"Test samples: {len(test_idxs)}")

    train_val_years_present = np.unique(times[valid_train_val].astype("datetime64[Y]").astype(int) + 1970)
    test_years_present = np.unique(times[valid_test].astype("datetime64[Y]").astype(int) + 1970)
    log(f"Train/val years represented: {train_val_years_present.tolist()}")
    log(f"Test years represented: {test_years_present.tolist()}")

    n_channels = int(metadata["n_vars"])
    n_outputs = int(target_values.shape[1])

    input_mean, input_std = get_or_create_normalization_stats(
        dataset_dir=dataset_dir,
        times=times,
        valid_train_val=valid_train_val,
        n_channels=n_channels,
        stats_path=args.normalization_stats,
        recompute=args.recompute_normalization,
        batch_size=args.batch_size,
    )

    log(f"Input mean (first 5): {input_mean.tolist()[:5]}")
    log(f"Input std (first 5): {input_std.tolist()[:5]}")

    train_dataset = WeatherTransferDataset(
        dataset_dir=dataset_dir,
        times=times,
        target_values=target_values,
        indices=train_idxs,
        input_mean=input_mean,
        input_std=input_std,
    )
    val_dataset = WeatherTransferDataset(
        dataset_dir=dataset_dir,
        times=times,
        target_values=target_values,
        indices=val_idxs,
        input_mean=input_mean,
        input_std=input_std,
    )
    test_dataset = WeatherTransferDataset(
        dataset_dir=dataset_dir,
        times=times,
        target_values=target_values,
        indices=test_idxs,
        input_mean=input_mean,
        input_std=input_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    device = torch.device(args.device)
    log(f"Using device: {device}")

    log(
        f"Building model {args.model_name} "
        f"(pretrained={args.pretrained}, n_input_channels={n_channels}, n_outputs={n_outputs})"
    )
    model = get_pretrained_backbone(
        model_name=args.model_name,
        n_outputs=n_outputs,
        n_input_channels=n_channels,
        pretrained=args.pretrained,
    )

    if args.weights_path is not None:
        log(f"Loading model weights from {args.weights_path}")
        model = load_model_weights(model, str(args.weights_path), device=device)

    pretrain_tag = "pretrained" if args.pretrained else "scratch"

    if args.train_mode == "gradual_unfreeze":
        mode_tag = f"{args.train_mode}_every{args.unfreeze_every}"
    else:
        mode_tag = args.train_mode

    stem_name = f"{args.model_name}_weather_{pretrain_tag}_{mode_tag}"

    best_model_path = args.output_dir / f"{stem_name}_best.pth"
    final_model_path = args.output_dir / f"{stem_name}_final.pth"
    history_path = args.output_dir / f"{stem_name}_metrics.json"

    log(f"Best model path: {best_model_path}")
    log(f"Final model path: {final_model_path}")
    log(f"Metrics JSON path: {history_path}")

    metrics_history = train_transfer_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_mode=args.train_mode,
        unfreeze_every=args.unfreeze_every,
        best_model_path=best_model_path,
        final_model_path=final_model_path,
        history_path=history_path,
        early_stopping_patience=args.early_stopping_patience,
    )

    if not args.skip_test:
        log("Evaluating best available model on test set...")
        criterion = nn.MSELoss()

        if best_model_path.exists():
            model = load_model_weights(model, str(best_model_path), device=device)
            log(f"Loaded best model weights from {best_model_path} for test evaluation")
        elif final_model_path.exists():
            model = load_model_weights(model, str(final_model_path), device=device)
            log(f"Loaded final model weights from {final_model_path} for test evaluation")
        else:
            log("Warning: No saved model weights found; evaluating current in-memory model")

        test_metrics = evaluate_model(
            model=model.to(device),
            dataloader=test_loader,
            device=device,
            criterion=criterion,
        )

        log(f"Test loss: {test_metrics['loss']:.6f}")
        log(f"Test mean RMSE: {test_metrics['rmse_mean']:.6f}")
        if "rmse_apcp_rain" in test_metrics:
            log(f"Test APCP>2mm RMSE: {test_metrics['rmse_apcp_rain']:.6f}")
        if "auc_apcp" in test_metrics:
            log(f"Test APCP AUC: {test_metrics['auc_apcp']:.6f}")

        metrics_history.append({"epoch": "test", **test_metrics})
        history_path.write_text(json.dumps(metrics_history, indent=2))
        log(f"Updated metrics history with test metrics at {history_path}")
    else:
        log("Skipping test evaluation as requested.")


if __name__ == "__main__":
    main()