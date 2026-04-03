import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # faster, non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import get_model


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PART2_OUTPUTS = PROJECT_ROOT / "part2_outputs"
SUMMARY_JSON_PATH = PART2_OUTPUTS / "confidence_analysis_summary.json"

DATASET_ROOT = Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset")
METADATA_PATH = DATASET_ROOT / "metadata.pt"
TARGETS_PATH = DATASET_ROOT / "targets.pt"

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "feature_saliency_2panel_gifs_fast"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If needed, manually set this
MANUAL_WEIGHTS_PATH = None
MANUAL_WEIGHTS_PATH = PROJECT_ROOT / "my_cnn_weights.pth"


# ============================================================
# Settings
# ============================================================

WINDOW = 24
FPS = 4
APCP_INDEX = 5
EPS = 1e-8

# Keep this small for speed. Set to None for all channels, but that will be much slower.
FEATURE_INDICES = [0, 1, 2, 3, 4, 6]

# Rendering speed knobs
FIGSIZE = (8, 4)
DPI = 120
USE_COLORBARS = False   # False is faster
ABS_SALIENCY = True


# ============================================================
# Loading helpers
# ============================================================

def load_summary():
    if not SUMMARY_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing summary JSON: {SUMMARY_JSON_PATH}")
    with open(SUMMARY_JSON_PATH, "r") as f:
        return json.load(f)


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")
    return torch.load(METADATA_PATH, map_location="cpu", weights_only=False)


def load_targets():
    if not TARGETS_PATH.exists():
        raise FileNotFoundError(f"Missing targets file: {TARGETS_PATH}")
    return torch.load(TARGETS_PATH, map_location="cpu", weights_only=False)


def get_variable_names(metadata):
    if isinstance(metadata, dict):
        if "variable_names" in metadata:
            return list(metadata["variable_names"])
        if "variables" in metadata:
            return list(metadata["variables"])
        if "n_vars" in metadata:
            return [f"feature_{i}" for i in range(int(metadata["n_vars"]))]
    return None


def resolve_weights_path(payload):
    if MANUAL_WEIGHTS_PATH is not None:
        return Path(MANUAL_WEIGHTS_PATH)

    weights_path = payload.get("weights_path")
    if weights_path is None:
        raise FileNotFoundError(
            "No weights_path found in confidence_analysis_summary.json and MANUAL_WEIGHTS_PATH is None."
        )

    weights_path = Path(weights_path)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    return weights_path


def load_model(metadata, payload):
    model = get_model(metadata)
    weights_path = resolve_weights_path(payload)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    print(f"Loading weights from: {weights_path}")
    state = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict):
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    else:
        raise ValueError(f"Unexpected checkpoint format in {weights_path}")

    model.eval()
    return model


# ============================================================
# Tensor helpers
# ============================================================

def normalize_map(x):
    x = x.float()
    return (x - x.min()) / (x.max() - x.min() + EPS)


def sanitize_filename(name):
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    out = str(name)
    for ch in bad_chars:
        out = out.replace(ch, "_")
    return out


def choose_feature_indices(feature_indices, n_channels):
    if feature_indices is None:
        return list(range(n_channels))
    return [i for i in feature_indices if 0 <= i < n_channels]


def load_input_tensor_by_index(times, t_idx):
    if t_idx < 0 or t_idx >= len(times):
        return None, None

    dt = pd.Timestamp(times[t_idx])
    x_path = DATASET_ROOT / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

    if not x_path.exists():
        return None, None

    x = torch.load(x_path, map_location="cpu", weights_only=False)

    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {tuple(x.shape)}")
        x = x[0]

    if x.ndim != 3:
        raise ValueError(f"Expected input shape (H, W, C), got {tuple(x.shape)}")

    return x.float(), dt


def compute_saliency_once(model, x_3d, target_index=APCP_INDEX):
    """
    x_3d: (H, W, C)
    returns:
      saliency: (H, W, C)
      pred_vec: (n_outputs,)
    """
    x = x_3d.unsqueeze(0).clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    output = model(x)
    output[0, target_index].backward()

    saliency = x.grad[0].detach().cpu()
    pred_vec = output.detach().cpu()[0]
    return saliency, pred_vec


# ============================================================
# Rendering
# ============================================================

def make_two_panel_frame(raw_map, saliency_map, feature_name, dt, offset, pred_apcp):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)

    raw_np = raw_map.numpy()
    sal_np = saliency_map.numpy()

    im0 = axes[0].imshow(raw_np, aspect="auto", cmap="viridis")
    axes[0].set_title(f"Raw: {feature_name}", fontsize=10)
    axes[0].axis("off")

    im1 = axes[1].imshow(sal_np, aspect="auto", cmap="hot")
    axes[1].set_title(f"Saliency: {feature_name}", fontsize=10)
    axes[1].axis("off")

    if USE_COLORBARS:
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"time={dt} | t{offset:+d}h | pred_apcp={pred_apcp:.3f}",
        fontsize=11
    )
    fig.tight_layout()

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    image = image[:, :, :3]
    plt.close(fig)
    return image


# ============================================================
# Core logic
# ============================================================

def precompute_window_data(model, times, center_t_idx, feature_indices):
    """
    Fast path:
    For each timepoint in the window, load x and compute saliency ONCE.
    Then store normalized raw/saliency maps only for requested channels.
    """
    cached = []

    for offset in range(-WINDOW, WINDOW + 1):
        idx = center_t_idx + offset

        x, dt = load_input_tensor_by_index(times, idx)
        if x is None:
            continue

        saliency, pred_vec = compute_saliency_once(model, x, target_index=APCP_INDEX)
        pred_apcp = float(pred_vec[APCP_INDEX].item())

        feature_payload = {}
        for ch in feature_indices:
            raw_map = normalize_map(x[:, :, ch].cpu())
            sal_map = saliency[:, :, ch].cpu()
            if ABS_SALIENCY:
                sal_map = sal_map.abs()
            sal_map = normalize_map(sal_map)

            feature_payload[ch] = {
                "raw": raw_map,
                "saliency": sal_map,
            }

        cached.append({
            "offset": offset,
            "timestamp": str(dt),
            "dt": dt,
            "pred_apcp": pred_apcp,
            "features": feature_payload,
        })

    return cached


def create_fast_gifs_for_example(example, example_label, model, times, variable_names):
    t_idx = int(example["t_idx"])
    example_name = f"{example_label}_tidx_{t_idx}"

    print(f"\nProcessing {example_name}")

    center_x, center_dt = load_input_tensor_by_index(times, t_idx)
    if center_x is None:
        print(f"  Missing center input for t_idx={t_idx}, skipping.")
        return

    _, _, n_channels = center_x.shape
    feature_indices = choose_feature_indices(FEATURE_INDICES, n_channels)

    if not feature_indices:
        print("  No valid feature indices, skipping.")
        return

    example_dir = OUTPUT_DIR / example_name
    example_dir.mkdir(parents=True, exist_ok=True)

    # Fast precompute: one model backward per timepoint
    print("  Precomputing raw maps + saliency once per timepoint...")
    cached_window = precompute_window_data(model, times, t_idx, feature_indices)
    print(f"  Cached {len(cached_window)} frames")

    manifest = {
        "example_name": example_name,
        "t_idx": t_idx,
        "timestamp_center": str(center_dt),
        "window": WINDOW,
        "fps": FPS,
        "apcp_target_index": APCP_INDEX,
        "feature_indices": feature_indices,
        "n_cached_frames": len(cached_window),
        "features": [],
    }

    for ch in feature_indices:
        feature_name = (
            variable_names[ch] if variable_names is not None and ch < len(variable_names)
            else f"channel_{ch:03d}"
        )
        safe_feature_name = sanitize_filename(feature_name)

        print(f"  Building GIF for feature {ch}: {feature_name}")

        frames = []
        frame_records = []

        for item in cached_window:
            raw_map = item["features"][ch]["raw"]
            sal_map = item["features"][ch]["saliency"]

            frame = make_two_panel_frame(
                raw_map=raw_map,
                saliency_map=sal_map,
                feature_name=feature_name,
                dt=item["dt"],
                offset=item["offset"],
                pred_apcp=item["pred_apcp"],
            )
            frames.append(frame)

            frame_records.append({
                "offset_hours": item["offset"],
                "timestamp": item["timestamp"],
                "pred_apcp": item["pred_apcp"],
            })

        if not frames:
            print("    No frames available, skipping.")
            continue

        gif_path = example_dir / f"{ch:03d}_{safe_feature_name}_2panel_fast.gif"
        imageio.mimsave(gif_path, frames, fps=FPS)

        manifest["features"].append({
            "channel_index": ch,
            "feature_name": feature_name,
            "gif_path": str(gif_path),
            "n_frames": len(frames),
            "frames": frame_records,
        })

        print(f"    Saved: {gif_path}")

    with open(example_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    payload = load_summary()
    metadata = load_metadata()
    targets = load_targets()

    variable_names = get_variable_names(metadata)
    times = targets["time"]

    model = load_model(metadata, payload)

    all_examples = []
    for i, ex in enumerate(payload.get("confidently_correct", []), start=1):
        all_examples.append((ex, f"{i:02d}_confidently_correct"))

    for i, ex in enumerate(payload.get("confidently_incorrect", []), start=1):
        all_examples.append((ex, f"{i:02d}_confidently_incorrect"))

    print(f"Total examples: {len(all_examples)}")
    print(f"Selected features: {FEATURE_INDICES}")

    for ex, label in all_examples:
        create_fast_gifs_for_example(
            example=ex,
            example_label=label,
            model=model,
            times=times,
            variable_names=variable_names,
        )

    print(f"\nDone. Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()