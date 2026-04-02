import json
from pathlib import Path

path = Path("outputs/training_metrics.json")

if not path.exists():
    print("training_metrics.json not found yet")
else:
    with open(path, "r") as f:
        data = json.load(f)

    print(f"Found {len(data)} epochs\n")
    for row in data:
        print(f"Epoch: {row['epoch']}")
        print(f"  train_loss:        {row['train_loss']:.6f}")
        print(f"  val_rmse_mean:     {row['val_rmse_mean']:.6f}")
        print(f"  val_rmse_apcp:     {row['val_rmse_apcp_rain']}")
        print(f"  val_auc_apcp:      {row['val_auc_apcp']}")
        print(f"  val_rmse:          {row['val_rmse']}")
        print()