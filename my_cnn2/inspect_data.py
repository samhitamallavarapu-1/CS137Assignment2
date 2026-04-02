from pathlib import Path
import torch
import pandas as pd

DATASET_ROOT = Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset")

metadata = torch.load(DATASET_ROOT / "metadata.pt", weights_only=False)
targets = torch.load(DATASET_ROOT / "targets.pt", weights_only=False)

print("\n=== METADATA KEYS ===")
print(metadata.keys())

print("\n=== TARGETS KEYS ===")
print(targets.keys())

times = targets["time"]
values = targets["values"]
binary = targets["binary_label"]

print("\n=== BASIC SHAPES ===")
print("times shape:", len(times))
print("values shape:", values.shape)
print("binary shape:", binary.shape)

print("\n=== VARIABLES ===")
if "variable_names" in metadata:
    for i, v in enumerate(metadata["variable_names"]):
        print(i, v)

print("\n=== N_VARS ===")
print(metadata.get("n_vars"))

print("\n=== TIME RANGE ===")
print("start:", times[0])
print("end:", times[-1])

# Load one sample input
dt = pd.Timestamp(times[0])
sample_path = DATASET_ROOT / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

print("\n=== SAMPLE INPUT PATH ===")
print(sample_path)

x = torch.load(sample_path, weights_only=True).float()
print("\n=== SAMPLE INPUT ===")
print("shape:", x.shape)
print("dtype:", x.dtype)
print("finite:", torch.isfinite(x).all().item())
print("nan count:", torch.isnan(x).sum().item())

print("\n=== SAMPLE TARGET ===")
print(values[24])
print("binary label:", binary[24])