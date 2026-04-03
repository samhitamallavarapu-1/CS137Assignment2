import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PART2_OUTPUTS = PROJECT_ROOT / "part2_outputs"
SUMMARY_JSON_PATH = PART2_OUTPUTS / "confidence_analysis_summary.json"

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "per_feature_heatmaps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of top features (from CSV) to save as reference
TOP_N_FEATURES = 10

EPS = 1e-8


# ============================================================
# Helpers
# ============================================================

def load_summary():
    if not SUMMARY_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing JSON: {SUMMARY_JSON_PATH}")

    with open(SUMMARY_JSON_PATH, "r") as f:
        return json.load(f)


def normalize_map(x):
    x = x.float()
    return (x - x.min()) / (x.max() - x.min() + EPS)


def save_heatmap(heatmap, title, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.imshow(heatmap.numpy(), aspect="auto")
    plt.colorbar(label="Normalized saliency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def process_example(group_name, idx, ex):
    example_id = f"{idx:02d}_{group_name}_tidx_{ex['t_idx']}"
    print(f"\nProcessing {example_id}")

    saliency_path = PROJECT_ROOT / ex["saliency_tensor_path"]
    csv_path = PROJECT_ROOT / ex["csv_path"]

    if not saliency_path.exists():
        print(f"  ❌ Missing saliency tensor: {saliency_path}")
        return

    if not csv_path.exists():
        print(f"  ❌ Missing CSV: {csv_path}")
        return

    # --------------------------------------------------------
    # Load saliency tensor
    # --------------------------------------------------------
    saliency = torch.load(saliency_path, map_location="cpu", weights_only=False)

    if saliency.ndim != 4 or saliency.shape[0] != 1:
        raise ValueError(f"Unexpected saliency shape: {saliency.shape}")

    # Remove batch dimension
    saliency = saliency[0]  # (H, W, C)

    if saliency.ndim != 3:
        raise ValueError(f"Expected (H, W, C), got {saliency.shape}")

    H, W, C = saliency.shape
    print(f"  Shape after squeeze: (H={H}, W={W}, C={C})")

    # --------------------------------------------------------
    # Load feature CSV (for reference only)
    # --------------------------------------------------------
    feature_df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------
    example_dir = OUTPUT_DIR / group_name / example_id
    example_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Save ALL per-channel heatmaps
    # --------------------------------------------------------
    channel_dir = example_dir / "all_channels"
    channel_dir.mkdir(parents=True, exist_ok=True)

    for ch in range(C):
        heatmap = normalize_map(saliency[:, :, ch].abs())

        save_heatmap(
            heatmap,
            title=f"{example_id} | channel {ch}",
            path=channel_dir / f"channel_{ch:03d}.png"
        )

    print(f"  Saved {C} channel heatmaps")

    # --------------------------------------------------------
    # Save top features (REFERENCE ONLY)
    # --------------------------------------------------------
    if "feature" in feature_df.columns:
        top_df = feature_df.head(TOP_N_FEATURES)
        top_df.to_csv(example_dir / f"top_{TOP_N_FEATURES}_features.csv", index=False)

    # --------------------------------------------------------
    # Save metadata for traceability
    # --------------------------------------------------------
    info = {
        "example_id": example_id,
        "group": group_name,
        "t_idx": ex.get("t_idx"),
        "timestamp": ex.get("timestamp"),
        "tensor_shape_original": list(torch.load(saliency_path, map_location="cpu", weights_only=False).shape),
        "tensor_shape_used": [H, W, C],
        "note": "Channels are NOT guaranteed to match feature names because CSV was sorted.",
    }

    with open(example_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    payload = load_summary()

    correct = payload.get("confidently_correct", [])
    incorrect = payload.get("confidently_incorrect", [])

    print(f"Found {len(correct)} confidently_correct examples")
    print(f"Found {len(incorrect)} confidently_incorrect examples")

    # Process correct examples
    for i, ex in enumerate(correct, start=1):
        process_example("confidently_correct", i, ex)

    # Process incorrect examples
    for i, ex in enumerate(incorrect, start=1):
        process_example("confidently_incorrect", i, ex)

    print("\n✅ DONE")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()