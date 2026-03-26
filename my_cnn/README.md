# my_cnn Model for CS137 Assignment 2

This folder contains the CNN model and training script for Part 1 of the assignment.

## Overview

- **Model**: WeatherCNN - a 4-layer convolutional neural network predicting 6 continuous weather variables (TMP, SPFH, etc.) from spatial snapshots.
- **Binary Prediction**: The binary label for rainfall > 2mm is **derived from the predicted APCP amount** (thresholded at 2.0), not predicted directly by the model.
- **Data**: Uses 2018-2024 weather data only. Train/val split is 80/20 from valid samples; test is a separate 20% holdout.
- **Features**: NaN filtering, input normalization, early stopping, caching, progress bars.

## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- numpy, pandas, tqdm, sklearn (for AUC)

## Training

From the `my_cnn` directory:

```bash
python train.py \
    --dataset-root /cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset \
    --output-model my_cnn_weights.pth \
    --epochs 6 \
    --batch-size 64 \
    --lr 1e-3 \
    --seed 42
```

### Key Arguments

- `--dataset-root`: Path to dataset (default: `/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset`)
- `--output-model`: Output model weights file (default: `my_cnn_weights.pth`)
- `--epochs`: Number of training epochs (default: 6)
- `--batch-size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--seed`: Random seed (default: 42)
- `--skip-test`: Skip test evaluation after training
- `--early-stopping-patience`: Epochs to wait for improvement (default: 3)
- `--no-early-stopping`: Disable early stopping
- `--rescan-nan`: Force rescan for NaN values and recompute valid indices (ignore cache)
- `--skip-normalization`: Skip input normalization (use mean=0, std=1)
- `--device`: Device to use (default: cuda if available, else cpu)

## Data Processing

- **NaN Filtering**: Automatically filters out samples with NaN in inputs. Uses caching for speed; force rescan with `--rescan-nan`.
- **Normalization**: Computes mean/std from a subset of training data per channel. Can be skipped with `--skip-normalization`.
- **Splitting**: 80% train/val from 2018-2024, 20% test holdout. Shuffled with seed.

## Output Files

- `my_cnn_weights.pth`: Saved model weights (best by validation RMSE if early stopping enabled)
- `training_metrics.json`: JSON with per-epoch metrics (RMSE for 6 vars, APCP>2mm RMSE, AUC)
- `valid_indices_train_val.npy` & `valid_indices_test.npy`: Cached valid sample indices
- `normalization_stats.pt`: Saved normalization parameters

## Metrics

- **RMSE**: Root Mean Squared Error for each of the 6 continuous variables
- **APCP>2mm RMSE**: RMSE for APCP predictions where true APCP > 2mm
- **AUC**: Area Under Curve for derived binary (predicted APCP > 2.0 vs true binary labels)

## Evaluation

Test evaluation is included in `train.py` (unless `--skip-test` is used). For separate evaluation, use the provided `evaluate.py` script from the dataset.

## Notes

- Ensure sufficient memory (e.g., 32GB+ in Slurm) to avoid OOM kills during data loading/normalization.
- The model expects input shape (450, 449, 7) and outputs (6,) continuous predictions.
- Binary labels are used only for AUC computation, derived from predictions.
