#!/bin/bash
#SBATCH --job-name=cars_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PYTHON="$HOME/.conda/envs/cs137-cnn/bin/python"

MODEL_NAME="resnet152"
STRATEGY="sliding_window"

echo "===================================="
echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Model: $MODEL_NAME"
echo "Strategy: $STRATEGY"
echo "===================================="

echo "Using Python: $PYTHON"
$PYTHON --version

echo "=== GPU INFO ==="
nvidia-smi || true

echo "=== TORCH INFO ==="
$PYTHON -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())"

echo "=== START TRAINING ==="

$PYTHON cars_train.py \
  --cars-root stanford_cars \
  --output-dir outputs \
  --model-name resnet152\
  --strategy sliding_window \
  --epochs 80 \
  --slide-every 10 \
  --window-size 2 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4

echo "===================================="
echo "Finished at $(date)"
echo "===================================="