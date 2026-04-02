from pathlib import Path
import shutil


def main():
    project_root = Path(__file__).resolve().parent
    outputs_dir = project_root / "outputs"

    weights_path = outputs_dir / "my_cnn_weights.pth"
    norm_path = outputs_dir / "normalization_stats.pt"

    if not weights_path.exists():
        raise FileNotFoundError("Weights not found. Train first.")

    if not norm_path.exists():
        raise FileNotFoundError("Normalization stats not found. Train first.")

    # Target evaluation folder (adjust if needed)
    eval_dir = Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/evaluation/smna")

    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation folder not found: {eval_dir}")

    # Copy files
    shutil.copy(weights_path, eval_dir / "my_cnn_weights.pth")
    shutil.copy(norm_path, eval_dir / "normalization_stats.pt")
    shutil.copy(project_root / "model.py", eval_dir / "model.py")

    print("\nExport complete:")
    print(f"  weights → {eval_dir / 'my_cnn_weights.pth'}")
    print(f"  norm    → {eval_dir / 'normalization_stats.pt'}")
    print(f"  model   → {eval_dir / 'model.py'}")


if __name__ == "__main__":
    main()