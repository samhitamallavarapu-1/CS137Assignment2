import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from model import get_model

APCP_INDEX = 5
DATASET_ROOT = Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset")
EXAMPLE_SELECTION_PATH = Path("outputs/saliency/example_selection.json")
OUTPUT_DIR = Path("outputs/saliency")
EPS = 1e-8


def load_metadata_and_targets():
    metadata = torch.load(DATASET_ROOT / "metadata.pt", weights_only=False)
    targets_data = torch.load(DATASET_ROOT / "targets.pt", weights_only=False)
    return metadata, targets_data


def load_input_tensor(times, t_idx):
    dt = pd.Timestamp(times[t_idx])
    x_path = DATASET_ROOT / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    x = torch.load(x_path, weights_only=True).float()
    return x, x_path, dt


def compute_saliency(model, x, target_index=APCP_INDEX):
    """
    x: (1, H, W, C)
    returns: (H, W, C)
    """
    x = x.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    output = model(x)
    target = output[0, target_index]
    target.backward()

    saliency = x.grad[0].detach().cpu()  # (H, W, C)
    return saliency, output.detach().cpu()[0]


def aggregate_spatial_saliency(saliency):
    """
    (H, W, C) -> (H, W)
    """
    return saliency.abs().mean(dim=-1)


def aggregate_channel_saliency(saliency):
    """
    (H, W, C) -> (C,)
    """
    return saliency.abs().mean(dim=(0, 1))


def normalize_2d_map(x):
    x = x.float()
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + EPS)


def save_heatmap_image(heatmap, title, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.numpy(), aspect="auto")
    plt.colorbar(label="Normalized saliency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_channel_barplot(channel_scores, variable_names, title, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "variable": variable_names,
        "saliency": channel_scores.numpy(),
    }).sort_values("saliency", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df)), df["saliency"].values)
    plt.xticks(range(len(df)), df["variable"].values, rotation=90)
    plt.ylabel("Mean absolute saliency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return df


def load_selected_examples():
    if not EXAMPLE_SELECTION_PATH.exists():
        raise FileNotFoundError(
            f"Missing example selection file: {EXAMPLE_SELECTION_PATH}. "
            f"Run part2_select_examples.py first."
        )

    with open(EXAMPLE_SELECTION_PATH, "r") as f:
        payload = json.load(f)

    return payload["selected_examples"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata, targets_data = load_metadata_and_targets()
    times = targets_data["time"]
    target_values = targets_data["values"]
    binary_labels = targets_data["binary_label"]

    variable_names = metadata.get(
        "variable_names",
        [f"feature_{i}" for i in range(int(metadata["n_vars"]))],
    )

    model = get_model(metadata)
    model.eval()

    selected_examples = load_selected_examples()

    summary_rows = []

    for case_name, ex in selected_examples.items():
        if ex is None:
            print(f"Skipping {case_name}: no example available")
            continue

        t_idx = int(ex["t_idx"])

        x, x_path, dt = load_input_tensor(times, t_idx)
        x_batch = x.unsqueeze(0)  # (1, H, W, C)

        saliency, pred_vec = compute_saliency(model, x_batch, target_index=APCP_INDEX)
        heatmap = aggregate_spatial_saliency(saliency)
        heatmap_norm = normalize_2d_map(heatmap)

        channel_scores = aggregate_channel_saliency(saliency)
        channel_df = save_channel_barplot(
            channel_scores=channel_scores,
            variable_names=variable_names,
            title=f"{case_name}: Channel Saliency for APCP Prediction",
            output_path=OUTPUT_DIR / f"{case_name}_channel_saliency.png",
        )
        channel_df.to_csv(OUTPUT_DIR / f"{case_name}_channel_saliency.csv", index=False)

        save_heatmap_image(
            heatmap=heatmap_norm,
            title=f"{case_name}: Spatial Saliency for APCP Prediction",
            output_path=OUTPUT_DIR / f"{case_name}_spatial_heatmap.png",
        )

        torch.save(saliency, OUTPUT_DIR / f"{case_name}_saliency.pt")
        torch.save(heatmap, OUTPUT_DIR / f"{case_name}_heatmap.pt")
        torch.save(pred_vec, OUTPUT_DIR / f"{case_name}_pred_vector.pt")

        actual_vec = target_values[t_idx + 24].float().cpu()
        true_apcp = float(actual_vec[APCP_INDEX].item())
        pred_apcp = float(pred_vec[APCP_INDEX].item())
        true_rain = bool(binary_labels[t_idx + 24].item())
        pred_rain = bool(pred_apcp > 2.0)

        top10 = channel_df.head(10).to_dict(orient="records")

        summary_rows.append({
            "case": case_name,
            "t_idx": t_idx,
            "timestamp": str(dt),
            "input_path": str(x_path),
            "pred_apcp": pred_apcp,
            "true_apcp": true_apcp,
            "pred_rain": pred_rain,
            "true_rain": true_rain,
            "top_10_channels": top10,
            "spatial_heatmap_path": str(OUTPUT_DIR / f"{case_name}_spatial_heatmap.png"),
            "channel_saliency_plot_path": str(OUTPUT_DIR / f"{case_name}_channel_saliency.png"),
        })

        print(f"\nProcessed {case_name}")
        print(f"  timestamp:  {dt}")
        print(f"  pred_apcp:  {pred_apcp:.4f}")
        print(f"  true_apcp:  {true_apcp:.4f}")
        print(f"  pred_rain:  {pred_rain}")
        print(f"  true_rain:  {true_rain}")
        print(f"  saved heatmap + saliency outputs to {OUTPUT_DIR}")

    summary_path = OUTPUT_DIR / "part2_saliency_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()