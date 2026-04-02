from pathlib import Path
import torch
import pandas as pd

from model import get_model

DATASET_ROOT = Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset")

metadata = torch.load(DATASET_ROOT / "metadata.pt", weights_only=False)
targets = torch.load(DATASET_ROOT / "targets.pt", weights_only=False)

times = targets["time"]
dt = pd.Timestamp(times[0])
sample_path = DATASET_ROOT / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"

x = torch.load(sample_path, weights_only=True).float()  # (450, 449, 42)

model = get_model(metadata)
with torch.no_grad():
    y = model(x.unsqueeze(0))

print("input shape:", x.unsqueeze(0).shape)
print("output shape:", y.shape)
print("output:", y)