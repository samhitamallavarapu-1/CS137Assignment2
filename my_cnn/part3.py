import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
except ImportError as e:
    raise ImportError(
        "This script requires scikit-learn. Install it in your environment first."
    ) from e


# ============================================================
# Logging
# ============================================================

def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ============================================================
# Dataset helpers
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
        x = x.permute(2, 0, 1).contiguous()

        return x, y


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


# ============================================================
# Model construction
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
                else:
                    new_conv.weight[:, :3].copy_(old_conv.weight)
                    if n_input_channels > 3:
                        mean_channel = old_conv.weight.mean(dim=1, keepdim=True)
                        for c in range(3, n_input_channels):
                            new_conv.weight[:, c:c + 1].copy_(mean_channel)

            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        model.conv1 = new_conv
        return model

    if hasattr(model, "features") and len(list(model.features.children())) > 0:
        first_block = list(model.features.children())[0]

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
                    new_conv.weight[:, :3].copy_(old_conv.weight)
                    if n_input_channels > 3:
                        mean_channel = old_conv.weight.mean(dim=1, keepdim=True)
                        for c in range(3, n_input_channels):
                            new_conv.weight[:, c:c + 1].copy_(mean_channel)
            model.features[0] = new_conv
            return model

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
                    new_conv.weight[:, :3].copy_(old_conv.weight)
                    if n_input_channels > 3:
                        mean_channel = old_conv.weight.mean(dim=1, keepdim=True)
                        for c in range(3, n_input_channels):
                            new_conv.weight[:, c:c + 1].copy_(mean_channel)
            first_block[0] = new_conv
            model.features[0] = first_block
            return model

    raise ValueError("Could not adapt first conv layer for this architecture.")


def get_pretrained_backbone(
    model_name: str,
    n_outputs: int,
    n_input_channels: int,
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


# ============================================================
# Checkpoint discovery
# ============================================================

def parse_checkpoint_info(path: Path) -> Dict[str, str]:
    stem = path.stem
    if "_weather_" not in stem:
        raise ValueError(f"Unexpected checkpoint naming: {path.name}")

    architecture, rest = stem.split("_weather_", 1)
    parts = rest.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected checkpoint naming: {path.name}")

    pretrain_tag = parts[0]
    kind = parts[-1]
    method_tag = "_".join(parts[1:-1])

    return {
        "architecture": architecture,
        "pretrain_tag": pretrain_tag,
        "method_tag": method_tag,
        "kind": kind,
    }


def discover_checkpoints(root_dir: Path, checkpoint_kind: str = "best") -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)

    for path in sorted(root_dir.glob("outputs_transfer_*/*.pth")):
        try:
            info = parse_checkpoint_info(path)
        except ValueError:
            continue

        if info["kind"] != checkpoint_kind:
            continue

        grouped[info["architecture"]].append({
            "path": path,
            **info,
        })

    return grouped


# ============================================================
# Layer selection / activation extraction
# ============================================================

def get_layer_names_for_architecture(model_name: str) -> List[str]:
    model_name = model_name.lower()

    if model_name.startswith("resnet"):
        return ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]

    if model_name.startswith("densenet"):
        return [
            "features.conv0",
            "features.denseblock1",
            "features.denseblock2",
            "features.denseblock3",
            "features.denseblock4",
            "classifier",
        ]

    if model_name.startswith("efficientnet"):
        return [
            "features.0",
            "features.2",
            "features.4",
            "features.6",
            "features.8",
            "classifier",
        ]

    if model_name.startswith("mobilenet"):
        return [
            "features.0",
            "features.3",
            "features.6",
            "features.9",
            "features.12",
            "classifier",
        ]

    if model_name.startswith("vit_"):
        return [
            "encoder.layers.encoder_layer_0",
            "encoder.layers.encoder_layer_3",
            "encoder.layers.encoder_layer_6",
            "encoder.layers.encoder_layer_9",
            "encoder.ln",
            "heads.head",
        ]

    raise ValueError(f"No default layer list for architecture {model_name}")


def summarize_activation(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]

    if not torch.is_tensor(tensor):
        raise ValueError("Hook output is not a tensor")

    if tensor.ndim == 4:
        return tensor.mean(dim=(2, 3))

    if tensor.ndim == 3:
        return tensor.mean(dim=1)

    if tensor.ndim == 2:
        return tensor

    return tensor.view(tensor.shape[0], -1)


def get_named_module(model: nn.Module, target_name: str) -> nn.Module:
    named = dict(model.named_modules())
    if target_name not in named:
        raise KeyError(f"Layer {target_name} not found in model")
    return named[target_name]


def collect_representations(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    layer_names: List[str],
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    model.eval()

    collected = {name: [] for name in layer_names}
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            try:
                summary = summarize_activation(output).detach().cpu()
                collected[name].append(summary)
            except Exception:
                pass
        return hook

    for name in layer_names:
        module = get_named_module(model, name)
        hooks.append(module.register_forward_hook(make_hook(name)))

    seen = 0
    with torch.no_grad():
        for inputs, _targets in tqdm(dataloader, desc="Collecting activations", leave=False):
            inputs = inputs.to(device)
            batch_size = inputs.shape[0]

            if max_samples is not None and seen >= max_samples:
                break

            if max_samples is not None and seen + batch_size > max_samples:
                inputs = inputs[: max_samples - seen]
                batch_size = inputs.shape[0]

            _ = model(inputs)
            seen += batch_size

    for h in hooks:
        h.remove()

    final = {}
    for name in layer_names:
        if len(collected[name]) == 0:
            raise RuntimeError(f"No activations collected for layer {name}")
        final[name] = torch.cat(collected[name], dim=0).numpy()

    return final


# ============================================================
# Similarity metrics
# ============================================================

def center_rows(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = center_rows(X)
    Y = center_rows(Y)

    hsic = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, ord="fro")
    norm_y = np.linalg.norm(Y.T @ Y, ord="fro")

    denom = norm_x * norm_y
    if denom <= 0:
        return float("nan")
    return float(hsic / denom)


def cca_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    max_dim: int = 50,
    n_components: Optional[int] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same number of samples for CCA")

    X = center_rows(X)
    Y = center_rows(Y)

    n = X.shape[0]
    if n < 3:
        return {"cca_mean": float("nan"), "cca_max": float("nan"), "cca_first": float("nan"), "cca_ncomp": 0}

    max_allowed = min(n - 1, X.shape[1], Y.shape[1], max_dim)
    if max_allowed < 1:
        return {"cca_mean": float("nan"), "cca_max": float("nan"), "cca_first": float("nan"), "cca_ncomp": 0}

    if n_components is None:
        n_components = max_allowed
    else:
        n_components = min(n_components, max_allowed)

    pca_x = PCA(n_components=n_components)
    pca_y = PCA(n_components=n_components)

    Xr = pca_x.fit_transform(X)
    Yr = pca_y.fit_transform(Y)

    cca = CCA(n_components=n_components, max_iter=1000)
    Xc, Yc = cca.fit(Xr, Yr).transform(Xr, Yr)

    corrs = []
    for i in range(n_components):
        x_i = Xc[:, i]
        y_i = Yc[:, i]
        sx = np.std(x_i)
        sy = np.std(y_i)
        if sx < eps or sy < eps:
            corrs.append(np.nan)
            continue
        corrs.append(np.corrcoef(x_i, y_i)[0, 1])

    corrs = np.array(corrs, dtype=float)
    corrs = corrs[np.isfinite(corrs)]

    if corrs.size == 0:
        return {"cca_mean": float("nan"), "cca_max": float("nan"), "cca_first": float("nan"), "cca_ncomp": int(n_components)}

    return {
        "cca_mean": float(np.mean(corrs)),
        "cca_max": float(np.max(corrs)),
        "cca_first": float(corrs[0]),
        "cca_ncomp": int(n_components),
    }


# ============================================================
# Plotting / saving
# ============================================================

def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    output_path: Path,
    value_fmt: str = ".2f",
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.2), max(4, len(row_labels) * 0.8)))
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = "nan" if not np.isfinite(val) else format(val, value_fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pair_lines(
    layers: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(layers)), values, marker="o")
    plt.xticks(range(len(layers)), layers, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("Layer")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_triangular_method_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    output_path: Path,
    value_fmt: str = ".2f",
    mask_lower: bool = True,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = matrix.copy()

    if mask_lower:
        for i in range(display.shape[0]):
            for j in range(display.shape[1]):
                if i > j:
                    display[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), max(6, len(labels) * 1.0)))
    im = ax.imshow(display, aspect="equal")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = display[i, j]
            if np.isnan(val):
                text = ""
            else:
                text = format(val, value_fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# Helpers for non-redundant pair naming / matrices
# ============================================================

def canonical_pair(m1: str, m2: str):
    return tuple(sorted([m1, m2]))


def initialize_square_similarity_dict(method_names: List[str], layer_names: List[str]):
    data = {}
    for metric in ["cka", "cca"]:
        data[metric] = {
            layer: np.full((len(method_names), len(method_names)), np.nan, dtype=float)
            for layer in layer_names
        }
    return data


# ============================================================
# Main comparison logic
# ============================================================

def compare_architecture_methods(
    architecture: str,
    checkpoints: List[Dict],
    metadata: Dict,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    cca_max_dim: int,
    max_val_samples: Optional[int],
    save_pair_line_plots: bool = False,
):
    if len(checkpoints) < 2:
        log(f"Skipping {architecture}: need at least 2 methods, found {len(checkpoints)}")
        return

    n_outputs = int(metadata["n_outputs"]) if "n_outputs" in metadata else None
    if n_outputs is None:
        raise RuntimeError("metadata must include n_outputs when passed into compare_architecture_methods")

    n_input_channels = int(metadata["n_vars"])
    layer_names = get_layer_names_for_architecture(architecture)

    checkpoints = sorted(checkpoints, key=lambda x: x["method_tag"])

    log(f"Processing architecture {architecture} with methods: {[c['method_tag'] for c in checkpoints]}")
    log(f"Using layers: {layer_names}")

    reps_by_method = {}
    method_names = [c["method_tag"] for c in checkpoints]
    method_to_idx = {m: i for i, m in enumerate(method_names)}

    for ckpt in checkpoints:
        pretrained_flag = (ckpt["pretrain_tag"] == "pretrained")
        model = get_pretrained_backbone(
            model_name=architecture,
            n_outputs=n_outputs,
            n_input_channels=n_input_channels,
            pretrained=pretrained_flag,
        )
        state_dict = torch.load(ckpt["path"], map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        log(f"Collecting reps for {architecture} | {ckpt['method_tag']} | {ckpt['path'].name}")
        reps = collect_representations(
            model=model,
            dataloader=val_loader,
            device=device,
            layer_names=layer_names,
            max_samples=max_val_samples,
        )
        reps_by_method[ckpt["method_tag"]] = reps

    pair_rows = []
    summary_rows = []
    square_similarity = initialize_square_similarity_dict(method_names, layer_names)

    # Diagonal = self-similarity
    for layer_name in layer_names:
        for m in method_names:
            idx = method_to_idx[m]
            square_similarity["cka"][layer_name][idx, idx] = 1.0
            square_similarity["cca"][layer_name][idx, idx] = 1.0

    method_pairs = list(combinations(method_names, 2))

    cka_pair_matrix = np.full((len(method_pairs), len(layer_names)), np.nan, dtype=float)
    cca_pair_matrix = np.full((len(method_pairs), len(layer_names)), np.nan, dtype=float)
    row_labels = []

    for pair_idx, (m1_raw, m2_raw) in enumerate(method_pairs):
        m1, m2 = canonical_pair(m1_raw, m2_raw)
        row_labels.append(f"{m1} vs {m2}")

        cka_vals = []
        cca_vals = []

        for layer_idx, layer_name in enumerate(layer_names):
            X = reps_by_method[m1][layer_name]
            Y = reps_by_method[m2][layer_name]

            cka_val = linear_cka(X, Y)
            cca_stats = cca_similarity(X, Y, max_dim=cca_max_dim)
            cca_val = cca_stats["cca_mean"]

            cka_pair_matrix[pair_idx, layer_idx] = cka_val
            cca_pair_matrix[pair_idx, layer_idx] = cca_val

            i = method_to_idx[m1]
            j = method_to_idx[m2]
            square_similarity["cka"][layer_name][i, j] = cka_val
            square_similarity["cka"][layer_name][j, i] = cka_val
            square_similarity["cca"][layer_name][i, j] = cca_val
            square_similarity["cca"][layer_name][j, i] = cca_val

            cka_vals.append(cka_val)
            cca_vals.append(cca_val)

            pair_rows.append({
                "architecture": architecture,
                "method_1": m1,
                "method_2": m2,
                "layer": layer_name,
                "cka": cka_val,
                "cca_mean": cca_stats["cca_mean"],
                "cca_max": cca_stats["cca_max"],
                "cca_first": cca_stats["cca_first"],
                "cca_ncomp": cca_stats["cca_ncomp"],
                "n_samples": int(X.shape[0]),
                "dim_1": int(X.shape[1]),
                "dim_2": int(Y.shape[1]),
            })

        summary_rows.append({
            "architecture": architecture,
            "method_1": m1,
            "method_2": m2,
            "mean_cka_across_layers": float(np.nanmean(cka_vals)),
            "std_cka_across_layers": float(np.nanstd(cka_vals)),
            "mean_cca_across_layers": float(np.nanmean(cca_vals)),
            "std_cca_across_layers": float(np.nanstd(cca_vals)),
            "min_cka": float(np.nanmin(cka_vals)),
            "max_cka": float(np.nanmax(cka_vals)),
            "min_cca": float(np.nanmin(cca_vals)),
            "max_cca": float(np.nanmax(cca_vals)),
        })

        if save_pair_line_plots:
            plot_pair_lines(
                layers=layer_names,
                values=cka_vals,
                title=f"{architecture}: CKA by layer ({m1} vs {m2})",
                ylabel="Linear CKA",
                output_path=output_dir / architecture / "plots" / "pair_lines" / f"{m1}_vs_{m2}_cka_by_layer.png",
            )

            plot_pair_lines(
                layers=layer_names,
                values=cca_vals,
                title=f"{architecture}: CCA by layer ({m1} vs {m2})",
                ylabel="CCA (mean canonical corr)",
                output_path=output_dir / architecture / "plots" / "pair_lines" / f"{m1}_vs_{m2}_cca_by_layer.png",
            )

    arch_dir = output_dir / architecture
    arch_dir.mkdir(parents=True, exist_ok=True)

    save_csv(pair_rows, arch_dir / f"{architecture}_layerwise_similarity.csv")
    save_csv(summary_rows, arch_dir / f"{architecture}_summary_similarity.csv")

    with open(arch_dir / f"{architecture}_summary_similarity.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    # Original pair-by-layer summary heatmaps
    plot_heatmap(
        matrix=cka_pair_matrix,
        row_labels=row_labels,
        col_labels=layer_names,
        title=f"{architecture}: Linear CKA across unique method pairs",
        output_path=arch_dir / "plots" / f"{architecture}_cka_pairwise_heatmap.png",
    )

    plot_heatmap(
        matrix=cca_pair_matrix,
        row_labels=row_labels,
        col_labels=layer_names,
        title=f"{architecture}: CCA across unique method pairs",
        output_path=arch_dir / "plots" / f"{architecture}_cca_pairwise_heatmap.png",
    )

    # New: non-redundant triangular method x method heatmaps for each layer
    for layer_name in layer_names:
        safe_layer_name = layer_name.replace(".", "_")

        plot_triangular_method_heatmap(
            matrix=square_similarity["cka"][layer_name],
            labels=method_names,
            title=f"{architecture}: CKA ({layer_name})",
            output_path=arch_dir / "plots" / "triangular_by_layer" / f"{architecture}_{safe_layer_name}_cka_triangular.png",
        )

        plot_triangular_method_heatmap(
            matrix=square_similarity["cca"][layer_name],
            labels=method_names,
            title=f"{architecture}: CCA ({layer_name})",
            output_path=arch_dir / "plots" / "triangular_by_layer" / f"{architecture}_{safe_layer_name}_cca_triangular.png",
        )

    log(f"Saved similarity analysis for {architecture} to {arch_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare CKA and CCA across transfer-learning training methods for the same architecture."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("/cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment2/my_cnn"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset"),
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path("/cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment2/my_cnn/normalization_stats_transfer.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("representation_similarity_outputs"),
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="all",
        help="Specific architecture to analyze, or 'all'",
    )
    parser.add_argument(
        "--checkpoint-kind",
        type=str,
        default="best",
        choices=["best", "final"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rescan-nan", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cca-max-dim", type=int, default=50)
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=512,
        help="Max number of validation samples to use for representation extraction. Use 0 for all.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional subset of method tags to compare, e.g. last_layer full gradual_unfreeze_every2",
    )
    parser.add_argument(
        "--save-pair-line-plots",
        action="store_true",
        help="Also save individual pairwise line plots. Leave off to reduce redundant outputs.",
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

    metadata["n_outputs"] = int(target_values.shape[1])

    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    _train_idxs, val_idxs, _test_idxs = get_validation_indices(
        dataset_dir=dataset_dir,
        times=times,
        target_values=target_values,
        binary_labels=binary_labels,
        seed=args.seed,
        rescan_nan=args.rescan_nan,
        cache_dir=cache_dir,
    )

    n_channels = int(metadata["n_vars"])
    input_mean, input_std = load_normalization_stats(args.normalization_stats, n_channels)

    val_dataset = WeatherTransferDataset(
        dataset_dir=dataset_dir,
        times=times,
        target_values=target_values,
        indices=val_idxs,
        input_mean=input_mean,
        input_std=input_std,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    device = torch.device(args.device)
    log(f"Using device: {device}")

    grouped = discover_checkpoints(args.root_dir, checkpoint_kind=args.checkpoint_kind)

    if args.architecture != "all":
        grouped = {args.architecture: grouped.get(args.architecture, [])}

    if not grouped:
        raise RuntimeError("No matching checkpoints found.")

    max_val_samples = None if args.max_val_samples == 0 else args.max_val_samples

    for architecture, ckpts in grouped.items():
        if args.methods is not None:
            ckpts = [c for c in ckpts if c["method_tag"] in args.methods]

        if len(ckpts) < 2:
            log(f"Skipping {architecture}: fewer than 2 matching methods after filtering.")
            continue

        compare_architecture_methods(
            architecture=architecture,
            checkpoints=ckpts,
            metadata=metadata,
            val_loader=val_loader,
            device=device,
            output_dir=args.output_dir,
            cca_max_dim=args.cca_max_dim,
            max_val_samples=max_val_samples,
            save_pair_line_plots=args.save_pair_line_plots,
        )


if __name__ == "__main__":
    main()