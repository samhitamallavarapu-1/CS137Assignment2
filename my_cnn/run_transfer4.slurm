#!/bin/bash
#SBATCH --job-name=cnn_gpu
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

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Using Python: $PYTHON"

$PYTHON --version
nvidia-smi

$PYTHON -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())"
$PYTHON train_transfer.py \
  --model-name resnet50 \
  --epochs 80 \
  --batch-size 64 \
  --lr 1e-3 \
  --device cuda \
  --output-dir outputs_transfer_scratch

echo "Finished at $(date)"
