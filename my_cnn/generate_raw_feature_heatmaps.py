import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PART2_OUTPUTS = PROJECT_ROOT / "part2_outputs"
SUMMARY_JSON_PATH = PART2_OUTPUTS / "confidence_analysis_summary.json"

# Optional: set to your true metadata path if available
METADATA_PATH = Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset/metadata.pt")

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "raw_feature_heatmaps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8


# ============================================================
# Helpers
# ============================================================

def load_summary():
    if not SUMMARY_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing JSON: {SUMMARY_JSON_PATH}")
    with open(SUMMARY_JSON_PATH, "r") as f:
        return json.load(f)


def load_variable_names():
    """
    Try to load real feature names from metadata.pt.
    Returns list[str] or None.
    """
    if not METADATA_PATH.exists():
        print(f"Metadata not found at {METADATA_PATH}. Will use channel indices.")
        return None

    metadata = torch.load(METADATA_PATH, map_location="cpu", weights_only=False)

    if isinstance(metadata, dict):
        if "variable_names" in metadata:
            variable_names = metadata["variable_names"]
            print(f"Loaded {len(variable_names)} variable names from metadata.")
            return list(variable_names)

        if "variables" in metadata:
            variable_names = metadata["variables"]
            print(f"Loaded {len(variable_names)} variable names from metadata['variables'].")
            return list(variable_names)

    print("Could not find variable names in metadata. Will use channel indices.")
    return None


def normalize_map(x):
    x = x.float()
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + EPS)


def sanitize_filename(name):
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    out = str(name)
    for ch in bad_chars:
        out = out.replace(ch, "_")
    return out


def save_heatmap(heatmap, title, output_path, colorbar_label="Normalized value"):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.imshow(heatmap.numpy(), aspect="auto")
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def prepare_input_tensor(x):
    """
    Accepts:
      (H, W, C)
      (1, H, W, C)
    Returns:
      (H, W, C)
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(x)}")

    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for 4D tensor, got shape {tuple(x.shape)}")
        x = x[0]

    if x.ndim != 3:
        raise ValueError(f"Expected input shape (H, W, C), got {tuple(x.shape)}")

    return x.float()


def get_feature_name(variable_names, ch):
    if variable_names is None or ch >= len(variable_names):
        return f"channel_{ch:03d}"
    return str(variable_names[ch])


def process_example(group_name, idx, ex, variable_names):
    example_id = f"{idx:02d}_{group_name}_tidx_{ex['t_idx']}"
    print(f"\nProcessing {example_id}")

    input_path_str = ex.get("input_path")
    if input_path_str is None:
        print("  Missing input_path in JSON. Skipping.")
        return

    input_path = Path(input_path_str)
    if not input_path.exists():
        print(f"  Missing raw input tensor: {input_path}")
        return

    x = torch.load(input_path, map_location="cpu", weights_only=False)
    x = prepare_input_tensor(x)  # (H, W, C)

    H, W, C = x.shape
    print(f"  Raw input shape: (H={H}, W={W}, C={C})")

    example_dir = OUTPUT_DIR / group_name / example_id
    all_dir = example_dir / "all_features"
    all_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "example_id": example_id,
        "group": group_name,
        "t_idx": ex.get("t_idx"),
        "timestamp": ex.get("timestamp"),
        "input_path": str(input_path),
        "shape": [H, W, C],
        "true_apcp_value": ex.get("true_apcp_value"),
        "pred_apcp_value": ex.get("pred_apcp_value"),
        "true_label": ex.get("true_label"),
        "pred_label": ex.get("pred_label"),
        "confidence": ex.get("confidence"),
        "features": [],
    }

    for ch in range(C):
        feature_name = get_feature_name(variable_names, ch)
        feature_map = x[:, :, ch]
        feature_map_norm = normalize_map(feature_map)

        file_name = f"{ch:03d}_{sanitize_filename(feature_name)}.png"
        output_path = all_dir / file_name

        save_heatmap(
            heatmap=feature_map_norm,
            title=f"{example_id} | raw feature | {feature_name}",
            output_path=output_path,
            colorbar_label="Normalized raw value",
        )

        manifest["features"].append({
            "channel_index": ch,
            "feature_name": feature_name,
            "output_path": str(output_path),
            "raw_min": float(feature_map.min().item()),
            "raw_max": float(feature_map.max().item()),
            "raw_mean": float(feature_map.mean().item()),
            "raw_std": float(feature_map.std().item()),
        })

    with open(example_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Saved {C} raw feature heatmaps to {all_dir}")


def main():
    payload = load_summary()
    variable_names = load_variable_names()

    correct = payload.get("confidently_correct", [])
    incorrect = payload.get("confidently_incorrect", [])

    print(f"Found {len(correct)} confidently_correct examples")
    print(f"Found {len(incorrect)} confidently_incorrect examples")

    for i, ex in enumerate(correct, start=1):
        process_example("confidently_correct", i, ex, variable_names)

    for i, ex in enumerate(incorrect, start=1):
        process_example("confidently_incorrect", i, ex, variable_names)

    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()