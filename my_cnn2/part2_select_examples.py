import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import get_model

APCP_INDEX = 5
DATASET_ROOT = Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset")
OUTPUT_PATH = Path("outputs/saliency/example_selection.json")


def choose_indices(times, years):
    years_array = times.astype("datetime64[Y]").astype(int) + 1970
    mask = np.isin(years_array, years)
    idx = np.where(mask)[0]
    return idx[idx + 24 < len(times)]


def load_sample_x(dataset_dir, times, t_idx):
    dt = pd.Timestamp(times[t_idx])
    x_path = dataset_dir / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    x = torch.load(x_path, weights_only=True).float()
    return x, str(x_path), str(dt)


def main():
    metadata = torch.load(DATASET_ROOT / "metadata.pt", weights_only=False)
    targets_data = torch.load(DATASET_ROOT / "targets.pt", weights_only=False)

    times = targets_data["time"]
    target_values = targets_data["values"]
    binary_labels = targets_data["binary_label"]

    # Match current validation choice from train.py
    val_indices = choose_indices(times, [2022])

    model = get_model(metadata)
    model.eval()

    rows = []

    with torch.no_grad():
        for t_idx in val_indices:
            x, x_path, timestamp = load_sample_x(DATASET_ROOT, times, t_idx)

            if not torch.isfinite(x).all():
                continue

            pred = model(x.unsqueeze(0)).squeeze(0).cpu()
            actual = target_values[t_idx + 24].float().cpu()

            pred_apcp = float(pred[APCP_INDEX].item())
            true_apcp = float(actual[APCP_INDEX].item())

            pred_rain = pred_apcp > 2.0
            true_rain = true_apcp > 2.0

            if pred_rain and true_rain:
                case = "TP"
            elif pred_rain and not true_rain:
                case = "FP"
            elif (not pred_rain) and true_rain:
                case = "FN"
            else:
                case = "TN"

            rows.append({
                "t_idx": int(t_idx),
                "timestamp": timestamp,
                "input_path": x_path,
                "pred_apcp": pred_apcp,
                "true_apcp": true_apcp,
                "pred_rain": bool(pred_rain),
                "true_rain": bool(true_rain),
                "case": case,
                "confidence_from_threshold": float(abs(pred_apcp - 2.0)),
                "absolute_apcp_error": float(abs(pred_apcp - true_apcp)),
            })

    selected = {}
    for case in ["TP", "FP", "FN", "TN"]:
        case_rows = [r for r in rows if r["case"] == case]
        if not case_rows:
            selected[case] = None
            continue

        # Pick the sample farthest from decision threshold
        case_rows = sorted(case_rows, key=lambda r: r["confidence_from_threshold"], reverse=True)
        selected[case] = case_rows[0]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "n_candidates": len(rows),
                "selected_examples": selected,
            },
            f,
            indent=2,
        )

    print(f"Saved example selection to {OUTPUT_PATH}")
    for case, ex in selected.items():
        print(case, "->", None if ex is None else ex["timestamp"])


if __name__ == "__main__":
    main()