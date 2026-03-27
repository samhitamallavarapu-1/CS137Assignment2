from pathlib import Path
import torch
import numpy as np

from analysis import compute_input_saliency
from model import get_model

def main():
    # Load metadata however you already do in train.py
    metadata = torch.load("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset/metadata.pt", weights_only=False)

    model = get_model(metadata)

    # Example input tensor with shape (1, H, W, C)
    x = torch.load("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset/inputs/2018/X_2018010100.pt", weights_only=True).float()
    x = x.unsqueeze(0)  # add batch dimension

    result = compute_input_saliency(
        model=model,
        weights_path="my_cnn_weights.pth",
        input_tensor=x,
        target_index=5,   # e.g. APCP output
    )

    print("Model output:", result["output"])
    print("Per-feature saliency:", result["feature_scores"])
    print("Mean feature saliency across batch:", result["feature_scores_mean"])
    print("Full saliency map shape:", result["saliency"].shape)

if __name__ == "__main__":
    main()