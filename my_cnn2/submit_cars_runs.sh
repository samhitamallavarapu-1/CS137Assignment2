#!/bin/bash
set -euo pipefail

mkdir -p logs

echo "Submitting all 8 Stanford Cars experiments..."

# DenseNet121
sbatch --export=ALL,MODEL_NAME=densenet121,STRATEGY=scratch     run_cars.slurm
sbatch --export=ALL,MODEL_NAME=densenet121,STRATEGY=last_layer  run_cars.slurm
sbatch --export=ALL,MODEL_NAME=densenet121,STRATEGY=full        run_cars.slurm
sbatch --export=ALL,MODEL_NAME=densenet121,STRATEGY=gradual     run_cars.slurm

# ResNet152
sbatch --export=ALL,MODEL_NAME=resnet152,STRATEGY=scratch       run_cars.slurm
sbatch --export=ALL,MODEL_NAME=resnet152,STRATEGY=last_layer    run_cars.slurm
sbatch --export=ALL,MODEL_NAME=resnet152,STRATEGY=full          run_cars.slurm
sbatch --export=ALL,MODEL_NAME=resnet152,STRATEGY=gradual       run_cars.slurm

echo "All jobs submitted!"