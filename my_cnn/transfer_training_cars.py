import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import StanfordCars
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

class TransformSubset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_train_val_test_splits(
    dataset_root: Path,
    seed: int,
    val_fraction: float = 0.2,
):
    train_base = StanfordCars(root=str(dataset_root), split="train", download=True)
    test_base = StanfordCars(root=str(dataset_root), split="test", download=True)

    n_train_total = len(train_base)
    indices = np.arange(n_train_total)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_val = int(n_train_total * val_fraction)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_subset = Subset(train_base, train_indices.tolist())
    val_subset = Subset(train_base, val_indices.tolist())
    test_subset = test_base

    return train_subset, val_subset, test_subset


# ============================================================
# Normalization
# ============================================================

def compute_normalization_stats(dataset, batch_size=64, num_workers=0):
    """
    Compute per-channel mean/std for an image dataset returning tensors in [0,1].
    """
    if len(dataset) == 0:
        log("Warning: Empty dataset for normalization. Using ImageNet-like identity fallback.")
        return torch.zeros(3), torch.ones(3)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for images, _labels in tqdm(loader, desc="Computing normalization stats", leave=False):
        # images shape: (B, C, H, W)
        if not torch.isfinite(images).all():
            continue

        b, c, h, w = images.shape
        pixels = b * h * w
        total_pixels += pixels

        channel_sum += images.sum(dim=(0, 2, 3), dtype=torch.float64)
        channel_sum_sq += (images ** 2).sum(dim=(0, 2, 3), dtype=torch.float64)

    if total_pixels == 0:
        log("Warning: No valid pixels found for normalization. Using fallback.")
        return torch.zeros(3), torch.ones(3)

    mean = channel_sum / total_pixels
    var = (channel_sum_sq / total_pixels) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    std = torch.clamp(std, min=1e-6)

    return mean.float(), std.float()


def get_or_create_normalization_stats(
    train_subset_raw,
    stats_path: Path,
    recompute: bool,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if stats_path.exists() and not recompute:
        log(f"Loading cached normalization stats from {stats_path}")
        stats = torch.load(stats_path, weights_only=False)
        mean = stats["input_mean"].float()
        std = torch.clamp(stats["input_std"].float(), min=1e-6)
        return mean, std

    raw_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    dataset_for_stats = TransformSubset(train_subset_raw, transform=raw_transform)

    log("Computing normalization stats from Stanford Cars training split...")
    mean, std = compute_normalization_stats(
        dataset_for_stats,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    torch.save({"input_mean": mean, "input_std": std}, stats_path)
    log(f"Saved normalization stats to {stats_path}")

    return mean, std


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


def get_pretrained_backbone(
    model_name: str = "resnet50",
    n_outputs: int = 196,
    pretrained: bool = True,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name in {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}:
        constructor = getattr(tv_models, model_name)
        model = constructor(weights=_get_default_weights(model_name) if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_outputs)
        return model

    if model_name in {"densenet121", "densenet169", "densenet201"}:
        constructor = getattr(tv_models, model_name)
        model = constructor(weights=_get_default_weights(model_name) if pretrained else None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, n_outputs)
        return model

    if model_name in {"efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"}:
        constructor = getattr(tv_models, model_name)
        model = constructor(weights=_get_default_weights(model_name) if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, n_outputs)
        return model

    if model_name in {"mobilenet_v3_small", "mobilenet_v3_large"}:
        constructor = getattr(tv_models, model_name)
        model = constructor(weights=_get_default_weights(model_name) if pretrained else None)
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
    if hasattr(model, "fc") and hasattr(model, "layer4"):
        return [
            model.fc,
            model.layer4,
            model.layer3,
            model.layer2,
            model.layer1,
            nn.Sequential(model.conv1, model.bn1),
        ]

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

    if hasattr(model, "features") and hasattr(model, "classifier"):
        return [get_final_layer(model)] + list(model.features.children())[::-1]

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
    n_correct = 0
    n_top5_correct = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            if not torch.isfinite(logits).all():
                continue

            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                continue

            batch_size = inputs.shape[0]
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            preds = logits.argmax(dim=1)
            n_correct += (preds == targets).sum().item()

            top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
            n_top5_correct += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

    if n_samples == 0:
        raise RuntimeError("No valid evaluation samples found.")

    return {
        "loss": float(running_loss / n_samples),
        "accuracy_top1": float(n_correct / n_samples),
        "accuracy_top5": float(n_top5_correct / n_samples),
        "n_samples": int(n_samples),
    }


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
    criterion = nn.CrossEntropyLoss()

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
        n_train_correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [{train_mode}]", leave=False)

        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            if not torch.isfinite(logits).all():
                pbar.set_postfix({"loss": "nonfinite_pred"})
                continue

            loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                pbar.set_postfix({"loss": "nonfinite_loss"})
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = inputs.shape[0]
            running_loss += loss.item() * batch_size
            n_train_samples += batch_size
            n_train_correct += (logits.argmax(dim=1) == targets).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / max(1, n_train_samples)
        train_acc = n_train_correct / max(1, n_train_samples)

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
            "train_accuracy_top1": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy_top1": float(val_metrics["accuracy_top1"]),
            "val_accuracy_top5": float(val_metrics["accuracy_top5"]),
            "strategy": train_mode,
            "strategy_note": strategy_note,
            "n_train_samples": int(n_train_samples),
            "n_val_samples": int(val_metrics["n_samples"]),
        }

        metrics_history.append(epoch_metrics)

        log(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_acc={val_metrics['accuracy_top1']:.4f} | "
            f"val_top5={val_metrics['accuracy_top5']:.4f} | "
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
    parser = argparse.ArgumentParser(description="Transfer learning trainer for torchvision Stanford Cars")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./data"),
        help="Root directory for Stanford Cars download/cache",
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path("normalization_stats_stanford_cars.pt"),
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
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image resize dimension",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of Stanford Cars train split used as validation",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log("Building Stanford Cars train/val/test splits...")
    train_subset_raw, val_subset_raw, test_subset_raw = build_train_val_test_splits(
        dataset_root=args.dataset_root,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )

    n_outputs = len(train_subset_raw.dataset.classes)
    log(f"Stanford Cars classes: {n_outputs}")
    log(f"Train samples: {len(train_subset_raw)}")
    log(f"Val samples: {len(val_subset_raw)}")
    log(f"Test samples: {len(test_subset_raw)}")

    input_mean, input_std = get_or_create_normalization_stats(
        train_subset_raw=train_subset_raw,
        stats_path=args.normalization_stats,
        recompute=args.recompute_normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    log(f"Input mean: {input_mean.tolist()}")
    log(f"Input std: {input_std.tolist()}")

    image_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=input_mean.tolist(), std=input_std.tolist()),
    ])

    train_dataset = TransformSubset(train_subset_raw, transform=image_transform)
    val_dataset = TransformSubset(val_subset_raw, transform=image_transform)
    test_dataset = TransformSubset(test_subset_raw, transform=image_transform)

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
        f"(pretrained={args.pretrained}, n_outputs={n_outputs})"
    )
    model = get_pretrained_backbone(
        model_name=args.model_name,
        n_outputs=n_outputs,
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

    stem_name = f"{args.model_name}_stanford_cars_{pretrain_tag}_{mode_tag}"

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
        criterion = nn.CrossEntropyLoss()

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
        log(f"Test top-1 accuracy: {test_metrics['accuracy_top1']:.4f}")
        log(f"Test top-5 accuracy: {test_metrics['accuracy_top5']:.4f}")

        metrics_history.append({"epoch": "test", **test_metrics})
        history_path.write_text(json.dumps(metrics_history, indent=2))
        log(f"Updated metrics history with test metrics at {history_path}")
    else:
        log("Skipping test evaluation as requested.")


if __name__ == "__main__":
    main()