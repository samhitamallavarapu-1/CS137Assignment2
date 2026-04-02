import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cars_dataset import get_cars_datasets
from cars_models import get_model


OUTPUTS_DIR = Path("outputs")
ANALYSIS_DIR = Path("outputs/part3_analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# PART A: METRICS SUMMARY / PLOTS
# =========================================================

def find_metrics_files(outputs_dir: Path):
    return sorted(outputs_dir.glob("*_cars_*_metrics.json"))


def load_metrics(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def parse_run_name(path: Path):
    stem = path.stem
    if not stem.endswith("_metrics"):
        raise ValueError(f"Unexpected metrics filename: {path.name}")

    stem = stem[:-8]
    parts = stem.split("_")

    if len(parts) < 3 or parts[1] != "cars":
        raise ValueError(f"Unexpected metrics filename: {path.name}")

    model_name = parts[0]
    strategy = "_".join(parts[2:])
    return model_name, strategy


def get_best_validation_row(rows):
    epoch_rows = [r for r in rows if r.get("epoch") != "test"]
    if not epoch_rows:
        return None
    return max(epoch_rows, key=lambda r: r.get("val_accuracy_top1", float("-inf")))


def get_test_row(rows):
    for r in rows:
        if r.get("epoch") == "test":
            return r
    return None


def build_summary(metrics_files):
    summary_rows = []

    for path in metrics_files:
        rows = load_metrics(path)
        model_name, strategy = parse_run_name(path)

        best_val = get_best_validation_row(rows)
        test_row = get_test_row(rows)

        if best_val is None:
            continue

        summary_rows.append({
            "file": path.name,
            "model_name": model_name,
            "strategy": strategy,
            "best_val_epoch": best_val.get("epoch"),
            "best_val_loss": best_val.get("val_loss"),
            "best_val_top1": best_val.get("val_accuracy_top1"),
            "best_val_top5": best_val.get("val_accuracy_top5"),
            "final_train_loss": best_val.get("train_loss"),
            "test_loss": None if test_row is None else test_row.get("loss"),
            "test_top1": None if test_row is None else test_row.get("accuracy_top1"),
            "test_top5": None if test_row is None else test_row.get("accuracy_top5"),
        })

    df = pd.DataFrame(summary_rows)

    if not df.empty:
        df = df.sort_values(
            by=["test_top1", "best_val_top1"],
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return df


def save_summary(df: pd.DataFrame):
    csv_path = ANALYSIS_DIR / "part3_summary.csv"
    json_path = ANALYSIS_DIR / "part3_summary.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"Saved CSV summary -> {csv_path}")
    print(f"Saved JSON summary -> {json_path}")


def plot_metric(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, filename: str):
    if df.empty or metric_col not in df.columns:
        print(f"Skipping plot for {metric_col}")
        return

    plot_df = df.dropna(subset=[metric_col]).copy()
    if plot_df.empty:
        print(f"No non-null values for {metric_col}")
        return

    plot_df["label"] = plot_df["model_name"] + "\n" + plot_df["strategy"]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(plot_df)), plot_df[metric_col].values)
    plt.xticks(range(len(plot_df)), plot_df["label"].values, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    out_path = ANALYSIS_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot -> {out_path}")


def plot_grouped_by_model(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, filename: str):
    if df.empty or metric_col not in df.columns:
        print(f"Skipping grouped plot for {metric_col}")
        return

    plot_df = df.dropna(subset=[metric_col]).copy()
    if plot_df.empty:
        print(f"No non-null values for grouped plot {metric_col}")
        return

    models = sorted(plot_df["model_name"].unique())
    strategies = sorted(plot_df["strategy"].unique())

    x = list(range(len(strategies)))
    width = 0.35 if len(models) == 2 else 0.8 / max(1, len(models))

    plt.figure(figsize=(10, 6))

    for i, model in enumerate(models):
        vals = []
        for strategy in strategies:
            row = plot_df[(plot_df["model_name"] == model) & (plot_df["strategy"] == strategy)]
            if len(row) == 0:
                vals.append(float("nan"))
            else:
                vals.append(row.iloc[0][metric_col])

        offsets = [j + (i - (len(models) - 1) / 2) * width for j in x]
        plt.bar(offsets, vals, width=width, label=model)

    plt.xticks(x, strategies, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = ANALYSIS_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved grouped plot -> {out_path}")


def make_rankings_text(df: pd.DataFrame):
    txt_path = ANALYSIS_DIR / "part3_rankings.txt"

    lines = []
    lines.append("Part 3 Rankings\n")

    lines.append("By test top-1 accuracy:\n")
    if "test_top1" in df.columns and not df.empty:
        ranked = df.dropna(subset=["test_top1"]).sort_values("test_top1", ascending=False)
        for i, (_, row) in enumerate(ranked.iterrows(), start=1):
            test_top5 = row["test_top5"]
            test_top5_str = "nan" if pd.isna(test_top5) else f"{test_top5:.4f}"
            lines.append(
                f"{i}. {row['model_name']} | {row['strategy']} | "
                f"test_top1={row['test_top1']:.4f} | "
                f"test_top5={test_top5_str}"
            )

    lines.append("\nBy best validation top-1 accuracy:\n")
    if "best_val_top1" in df.columns and not df.empty:
        ranked = df.dropna(subset=["best_val_top1"]).sort_values("best_val_top1", ascending=False)
        for i, (_, row) in enumerate(ranked.iterrows(), start=1):
            best_val_top5 = row["best_val_top5"]
            best_val_top5_str = "nan" if pd.isna(best_val_top5) else f"{best_val_top5:.4f}"
            lines.append(
                f"{i}. {row['model_name']} | {row['strategy']} | "
                f"best_val_top1={row['best_val_top1']:.4f} | "
                f"best_val_top5={best_val_top5_str}"
            )

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved rankings -> {txt_path}")


# =========================================================
# PART B: CKA / CCA REPRESENTATION SIMILARITY
# =========================================================

def get_val_loader(cars_root: Path, batch_size: int, num_workers: int, seed: int, val_fraction: float):
    train_dataset_full, _test_dataset = get_cars_datasets(cars_root, image_size=224)

    indices = np.arange(len(train_dataset_full))
    labels = np.array([train_dataset_full.samples[i]["label"] for i in indices])

    _train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )

    val_dataset = Subset(train_dataset_full, val_idx)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return val_loader


def get_layer_names_for_architecture(model_name: str):
    model_name = model_name.lower()

    if model_name == "densenet121":
        return [
            "features.conv0",
            "features.denseblock1",
            "features.denseblock2",
            "features.denseblock3",
            "features.denseblock4",
            "classifier",
        ]

    if model_name == "resnet152":
        return [
            "conv1",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "fc",
        ]

    raise ValueError(f"Unsupported architecture for similarity analysis: {model_name}")


def get_named_module(model, target_name: str):
    named = dict(model.named_modules())
    if target_name not in named:
        raise KeyError(f"Layer {target_name} not found in model")
    return named[target_name]


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


def collect_representations(model, dataloader, device, layer_names, max_samples=None):
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
        for images, _labels in tqdm(dataloader, desc="Collecting activations", leave=False):
            images = images.to(device)

            batch_size = images.shape[0]
            if max_samples is not None and seen >= max_samples:
                break

            if max_samples is not None and seen + batch_size > max_samples:
                images = images[: max_samples - seen]
                batch_size = images.shape[0]

            _ = model(images)
            seen += batch_size

    for h in hooks:
        h.remove()

    final = {}
    for name in layer_names:
        if len(collected[name]) == 0:
            raise RuntimeError(f"No activations collected for layer {name}")
        final[name] = torch.cat(collected[name], dim=0).numpy()

    return final


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


def cca_similarity(X: np.ndarray, Y: np.ndarray, max_dim: int = 50, eps: float = 1e-8):
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same number of samples")

    X = center_rows(X)
    Y = center_rows(Y)

    n = X.shape[0]
    if n < 3:
        return {"cca_mean": float("nan"), "cca_max": float("nan"), "cca_first": float("nan"), "cca_ncomp": 0}

    n_components = min(n - 1, X.shape[1], Y.shape[1], max_dim)
    if n_components < 1:
        return {"cca_mean": float("nan"), "cca_max": float("nan"), "cca_first": float("nan"), "cca_ncomp": 0}

    pca_x = PCA(n_components=n_components)
    pca_y = PCA(n_components=n_components)

    Xr = pca_x.fit_transform(X)
    Yr = pca_y.fit_transform(Y)

    cca = CCA(n_components=n_components, max_iter=1000)
    Xc, Yc = cca.fit(Xr, Yr).transform(Xr, Yr)

    corrs = []
    for i in range(n_components):
        sx = np.std(Xc[:, i])
        sy = np.std(Yc[:, i])
        if sx < eps or sy < eps:
            corrs.append(np.nan)
        else:
            corrs.append(np.corrcoef(Xc[:, i], Yc[:, i])[0, 1])

    corrs = np.array(corrs, dtype=float)
    corrs = corrs[np.isfinite(corrs)]

    if corrs.size == 0:
        return {
            "cca_mean": float("nan"),
            "cca_max": float("nan"),
            "cca_first": float("nan"),
            "cca_ncomp": int(n_components),
        }

    return {
        "cca_mean": float(np.mean(corrs)),
        "cca_max": float(np.max(corrs)),
        "cca_first": float(corrs[0]),
        "cca_ncomp": int(n_components),
    }


def save_csv(rows, path: Path):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved CSV -> {path}")


def plot_heatmap(matrix, row_labels, col_labels, title, output_path, value_fmt=".2f"):
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

    print(f"Saved heatmap -> {output_path}")


def plot_pair_lines(layers, values, title, ylabel, output_path):
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

    print(f"Saved line plot -> {output_path}")


def get_checkpoint_path(model_name: str, strategy: str):
    return OUTPUTS_DIR / f"{model_name}_cars_{strategy}_best.pth"


def analyze_similarity_for_architecture(
    architecture: str,
    strategies,
    cars_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_fraction: float,
    device: torch.device,
    max_val_samples: int,
    cca_max_dim: int,
):
    # include scratch too
    compare_strategies = [s for s in strategies if s in {"scratch", "last_layer", "full", "gradual"}]
    if len(compare_strategies) < 2:
        print(f"Skipping similarity for {architecture}: need >=2 strategies.")
        return

    val_loader = get_val_loader(
        cars_root=cars_root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        val_fraction=val_fraction,
    )

    layer_names = get_layer_names_for_architecture(architecture)
    reps_by_strategy = {}

    for strategy in compare_strategies:
        ckpt_path = get_checkpoint_path(architecture, strategy)
        if not ckpt_path.exists():
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue

        pretrained = strategy != "scratch"
        model = get_model(
            model_name=architecture,
            num_classes=196,
            pretrained=pretrained,
        )
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        print(f"Collecting representations for {architecture} | {strategy}")
        reps = collect_representations(
            model=model,
            dataloader=val_loader,
            device=device,
            layer_names=layer_names,
            max_samples=max_val_samples,
        )
        reps_by_strategy[strategy] = reps

    available = sorted(reps_by_strategy.keys())
    if len(available) < 2:
        print(f"Not enough available strategies for {architecture}")
        return

    pair_rows = []
    summary_rows = []
    pair_names = list(combinations(available, 2))

    cka_matrix = np.full((len(pair_names), len(layer_names)), np.nan, dtype=float)
    cca_matrix = np.full((len(pair_names), len(layer_names)), np.nan, dtype=float)
    row_labels = []

    for pair_idx, (s1, s2) in enumerate(pair_names):
        row_labels.append(f"{s1} vs {s2}")
        cka_vals = []
        cca_vals = []

        for layer_idx, layer_name in enumerate(layer_names):
            X = reps_by_strategy[s1][layer_name]
            Y = reps_by_strategy[s2][layer_name]

            cka_val = linear_cka(X, Y)
            cca_stats = cca_similarity(X, Y, max_dim=cca_max_dim)

            cka_matrix[pair_idx, layer_idx] = cka_val
            cca_matrix[pair_idx, layer_idx] = cca_stats["cca_mean"]

            cka_vals.append(cka_val)
            cca_vals.append(cca_stats["cca_mean"])

            pair_rows.append({
                "architecture": architecture,
                "strategy_1": s1,
                "strategy_2": s2,
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
            "strategy_1": s1,
            "strategy_2": s2,
            "mean_cka_across_layers": float(np.nanmean(cka_vals)),
            "std_cka_across_layers": float(np.nanstd(cka_vals)),
            "mean_cca_across_layers": float(np.nanmean(cca_vals)),
            "std_cca_across_layers": float(np.nanstd(cca_vals)),
            "min_cka": float(np.nanmin(cka_vals)),
            "max_cka": float(np.nanmax(cka_vals)),
            "min_cca": float(np.nanmin(cca_vals)),
            "max_cca": float(np.nanmax(cca_vals)),
        })

        plot_pair_lines(
            layers=layer_names,
            values=cka_vals,
            title=f"{architecture}: CKA by layer ({s1} vs {s2})",
            ylabel="Linear CKA",
            output_path=ANALYSIS_DIR / architecture / "similarity_plots" / f"{s1}_vs_{s2}_cka_by_layer.png",
        )

        plot_pair_lines(
            layers=layer_names,
            values=cca_vals,
            title=f"{architecture}: CCA by layer ({s1} vs {s2})",
            ylabel="CCA (mean canonical corr)",
            output_path=ANALYSIS_DIR / architecture / "similarity_plots" / f"{s1}_vs_{s2}_cca_by_layer.png",
        )

    arch_dir = ANALYSIS_DIR / architecture
    arch_dir.mkdir(parents=True, exist_ok=True)

    save_csv(pair_rows, arch_dir / f"{architecture}_layerwise_similarity.csv")
    save_csv(summary_rows, arch_dir / f"{architecture}_summary_similarity.csv")

    with open(arch_dir / f"{architecture}_summary_similarity.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    plot_heatmap(
        matrix=cka_matrix,
        row_labels=row_labels,
        col_labels=layer_names,
        title=f"{architecture}: Linear CKA across strategies",
        output_path=arch_dir / "similarity_plots" / f"{architecture}_cka_heatmap.png",
    )

    plot_heatmap(
        matrix=cca_matrix,
        row_labels=row_labels,
        col_labels=layer_names,
        title=f"{architecture}: CCA across strategies",
        output_path=arch_dir / "similarity_plots" / f"{architecture}_cca_heatmap.png",
    )


# =========================================================
# MAIN
# =========================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Part 3 Cars results + representation similarity")
    parser.add_argument("--cars-root", type=Path, default=Path("stanford_cars"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--cca-max-dim", type=int, default=50)
    args = parser.parse_args()

    metrics_files = find_metrics_files(OUTPUTS_DIR)

    if not metrics_files:
        print("No Stanford Cars metrics files found.")
        return

    print(f"Found {len(metrics_files)} metrics files:")
    for p in metrics_files:
        print(f"  - {p.name}")

    df = build_summary(metrics_files)

    if df.empty:
        print("No valid summary rows could be built.")
        return

    print("\n=== SUMMARY ===")
    print(df)

    save_summary(df)

    plot_metric(
        df,
        metric_col="test_top1",
        title="Part 3: Test Top-1 Accuracy by Run",
        ylabel="Test Top-1 Accuracy",
        filename="test_top1_by_run.png",
    )
    plot_metric(
        df,
        metric_col="test_top5",
        title="Part 3: Test Top-5 Accuracy by Run",
        ylabel="Test Top-5 Accuracy",
        filename="test_top5_by_run.png",
    )
    plot_metric(
        df,
        metric_col="best_val_top1",
        title="Part 3: Best Validation Top-1 Accuracy by Run",
        ylabel="Best Validation Top-1 Accuracy",
        filename="best_val_top1_by_run.png",
    )
    plot_metric(
        df,
        metric_col="best_val_top5",
        title="Part 3: Best Validation Top-5 Accuracy by Run",
        ylabel="Best Validation Top-5 Accuracy",
        filename="best_val_top5_by_run.png",
    )

    plot_grouped_by_model(
        df,
        metric_col="test_top1",
        title="Part 3: Test Top-1 Accuracy by Architecture and Strategy",
        ylabel="Test Top-1 Accuracy",
        filename="grouped_test_top1.png",
    )
    plot_grouped_by_model(
        df,
        metric_col="test_top5",
        title="Part 3: Test Top-5 Accuracy by Architecture and Strategy",
        ylabel="Test Top-5 Accuracy",
        filename="grouped_test_top5.png",
    )

    make_rankings_text(df)

    device = torch.device(args.device)

    for architecture in sorted(df["model_name"].unique()):
        strategies = sorted(df[df["model_name"] == architecture]["strategy"].unique())
        print(f"\n=== Similarity analysis for {architecture} ===")
        analyze_similarity_for_architecture(
            architecture=architecture,
            strategies=strategies,
            cars_root=args.cars_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            val_fraction=args.val_fraction,
            device=device,
            max_val_samples=args.max_val_samples,
            cca_max_dim=args.cca_max_dim,
        )


if __name__ == "__main__":
    main()