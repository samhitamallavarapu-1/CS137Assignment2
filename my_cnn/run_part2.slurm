#!/bin/bash
#SBATCH --job-name=cnn_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=part2_logslogs/%x_%j.out
#SBATCH --error=part2_logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p part2_logs
mkdir -p analysis_outputs

PYTHON="$HOME/.conda/envs/cs137-cnn/bin/python"

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Working directory: $(pwd)"
echo "Using Python: $PYTHON"

$PYTHON --version

echo "Step 1: selecting examples..."
$PYTHON part2_select_examples.py

echo "Step 2: generating saliency outputs..."
$PYTHON part2_saliency.py

echo "Finished at $(date)"