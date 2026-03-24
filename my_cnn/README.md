# my_cnn model for CS137 Assignment 2

This folder contains the CNN model and training script for Part 1.

## Requirements

- Python 3.8+
- PyTorch (with CUDA if available)
- numpy, pandas, sklearn (optional for AUC), tqdm (optional)

## Training

From the repository root:

```bash
cd /cluster/tufts/c26sp1cs0137/smalla01/CS137Assignment2
python my_cnn/train.py \
    --dataset-root /cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset \
    --output-model my_cnn/best_model.pth \
    --epochs 6 \
    --batch-size 8 \
    --lr 1e-3
```

The script will train on year 2018, validate on year 2019, and test on year 2021.

## Output

- `my_cnn/best_model.pth`: best weights by validation RMSE
- `training_metrics.json`: per-epoch validation + test metrics

## Evaluation

The `train.py` script already includes test evaluation (RMSE + APCP >2mm RMSE + AUC).
