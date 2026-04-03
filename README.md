# CS137 Assignment 2: CNN Weather Forecasting and Transfer Learning

This repository contains implementations for CS137 Assignment 2, focusing on convolutional neural networks for weather forecasting and transfer learning on the Stanford Cars dataset. The assignment includes two main implementations (`my_cnn` and `my_cnn2`) that explore different approaches to the same problems.

## Directory Structure

```
CS137Assignment2/
├── assignment2.pdf                 # Assignment specification document
├── .git/                          # Git repository metadata
├── .gitignore                     # Git ignore rules
│
├── my_cnn/                        # First implementation
│   ├── Core Training Scripts
│   │   ├── train.py               # Weather CNN training
│   │   ├── transfer_training_cars.py  # Transfer learning on Stanford Cars
│   │   ├── transfer_training_weather.py  # Transfer learning for weather
│   │   └── model.py               # WeatherCNN model architecture
│   │
│   ├── Analysis & Visualization
│   │   ├── analysis.py            # Saliency and analysis functions
│   │   ├── part2.py               # Part 2: Input saliency analysis
│   │   ├── part2_saliency.py      # Saliency computation
│   │   ├── part2_select_examples.py  # Example selection
│   │   ├── part3.py               # Part 3: Research analysis
│   │   ├── visualization.py       # Visualization utilities
│   │   └── plot.py                # Plotting functions
│   │
│   ├── Slurm Scripts
│   │   ├── run_train.slurm        # Submit training job
│   │   ├── run_part2.slurm        # Submit saliency analysis
│   │   ├── run_part3.slurm        # Submit research analysis
│   │   └── run_transfer[1-8].slurm  # Transfer learning experiments
│   │
│   ├── Models & Data
│   │   ├── my_cnn_weights.pth     # Trained weather CNN weights
│   │   ├── normalization_stats*.pt  # Normalization statistics
│   │   ├── valid_indices_*.npy    # Pre-computed valid data indices
│   │   └── stanford_cars/         # Stanford Cars dataset
│   │
│   ├── Outputs & Results
│   │   ├── training_metrics.json  # Training metrics
│   │   ├── outputs_transfer_*/    # Transfer learning results
│   │   ├── analysis_outputs/      # Saliency analysis outputs
│   │   ├── plots/                 # Generated visualizations
│   │   ├── logs/                  # Training logs
│   │   └── representation_similarity_outputs/  # Feature similarity analysis
│   │
│   └── README.md                  # Detailed README for my_cnn
│
└── my_cnn2/                       # Second implementation
    ├── Core Scripts
    │   ├── train.py               # Weather CNN training (alternative)
    │   ├── cars_train.py          # Stanford Cars training with transfer learning
    │   ├── model.py               # WeatherCNN model
    │   ├── mycnn2_model.py        # Alternative WeatherCNN
    │   ├── mycnn3_model.py        # Another WeatherCNN variant
    │   └── utils.py               # Utility functions
    │
    ├── Dataset & Models
    │   ├── cars_dataset.py        # Stanford Cars dataset utilities
    │   ├── cars_models.py         # Transfer learning model utilities
    │   ├── inspect_data.py        # Data inspection scripts
    │   └── export_model.py        # Model export utilities
    │
    ├── Analysis
    │   ├── part2_saliency.py      # Saliency analysis
    │   ├── part2_select_examples.py  # Example selection
    │   ├── part3_analysis.py      # Part 3 analysis
    │   ├── peek_metrics.py        # Metrics inspection
    │   └── plots.py               # Plotting utilities
    │
    ├── Testing & Training
    │   ├── test_cars_dataset.py   # Test cars dataset
    │   ├── test_model.py          # Model testing
    │   ├── train_test.py          # Training tests
    │   ├── train_test2.py         # Additional training tests
    │   └── train.py               # Main training script
    │
    ├── Slurm Scripts
    │   ├── run_cars.slurm         # Submit cars training
    │   ├── run_weather.slurm      # Submit weather training
    │   ├── run_weather_cnn2.slurm # Submit CNN2 weather training
    │   ├── run_weather_cnn3.slurm # Submit CNN3 weather training
    │   └── submit_cars_runs.sh    # Batch submit cars runs
    │
    ├── Models & Metrics
    │   ├── my_cnn2_weights.pth    # CNN2 trained weights
    │   ├── my_cnn3_weights.pth    # CNN3 trained weights
    │   ├── my_cnn2_training_metrics.json  # CNN2 metrics
    │   ├── my_cnn3_training_metrics.json  # CNN3 metrics
    │   ├── normalization_stats.pt # Normalization stats
    │   └── commandline_check.txt  # Command line outputs
    │
    └── Output Directories
        ├── logs/                  # Training logs
        └── outputs/               # Training outputs
```

## Project Overview

This assignment explores convolutional neural networks for two main tasks:

### 1. Weather Forecasting CNN
**Objective:** Predict future weather conditions (24 hours ahead) from spatial weather maps.

**Input:** Historical weather data with multiple channels (temperature, pressure, humidity, etc.) in spatial grids.

**Output:** 6 regression targets representing future weather conditions.

**Models:** Custom WeatherCNN architectures with convolutional encoder and fully-connected head.

### 2. Transfer Learning on Stanford Cars
**Objective:** Apply transfer learning techniques to classify car models from the Stanford Cars dataset.

**Dataset:** Stanford Cars - fine-grained classification of car makes and models.

**Pre-trained Models:** ResNet50, DenseNet121 from torchvision.

**Fine-tuning Strategies:**
- **Scratch:** Train from random initialization
- **Last Layer:** Fine-tune only the final classification layer
- **Gradual:** Progressive unfreezing of layers during training
- **Full:** Fine-tune all layers

### Assignment Parts

#### Part 1: Weather CNN Training
- Implement and train WeatherCNN on weather forecasting task
- Evaluate performance and save trained models

#### Part 2: Input Saliency Analysis
- Compute saliency maps to understand which input features are most important
- Analyze spatial and channel-wise importance for weather predictions

#### Part 3: Research Question Analysis
- Investigate specific research questions about model behavior
- May include representation similarity, transfer learning analysis, etc.

## Implementations

### my_cnn/
The primary implementation with comprehensive analysis and visualization tools. Includes detailed saliency analysis, multiple transfer learning experiments, and extensive logging.

**Key Features:**
- Modular code structure
- Extensive analysis tools (saliency, visualization)
- Multiple transfer learning configurations
- HPC Slurm job scripts for cluster computing

### my_cnn2/
An alternative implementation with different model variants and testing utilities. Focuses on both weather and cars training with additional model architectures.

**Key Features:**
- Multiple WeatherCNN variants (CNN2, CNN3)
- Comprehensive testing scripts
- Batch job submission scripts
- Model export and inspection tools

## Usage

### Environment Setup

**Dependencies:**
- PyTorch >= 1.9.0
- TorchVision >= 0.10.0
- NumPy, Pandas
- Matplotlib, Scipy
- tqdm, scikit-learn
- Pillow

**HPC Environment:**
- Slurm job scheduler
- GPU resources (CUDA)
- Access to weather dataset and Stanford Cars

### Training Weather CNN

#### Using my_cnn/
```bash
cd my_cnn
python train.py --data_dir /path/to/weather/data --output_dir ./logs
# Or submit to HPC:
sbatch run_train.slurm
```

#### Using my_cnn2/
```bash
cd my_cnn2
python train.py --data_dir /path/to/weather/data
# Or submit to HPC:
sbatch run_weather.slurm
```

### Transfer Learning on Stanford Cars

#### Using my_cnn/
```bash
cd my_cnn
python transfer_training_cars.py \
  --data_dir ./stanford_cars \
  --output_dir ./outputs_transfer_cars_full_resnet50 \
  --model_name resnet50 \
  --fine_tune_method full
# Or submit individual experiments:
sbatch run_transfer1.slurm  # Scratch training
sbatch run_transfer2.slurm  # Last layer fine-tuning
# ... etc.
```

#### Using my_cnn2/
```bash
cd my_cnn2
python cars_train.py \
  --data_dir /path/to/stanford_cars \
  --output_dir ./outputs \
  --model_name resnet50 \
  --fine_tune_method full
# Or submit to HPC:
sbatch run_cars.slurm
```

### Running Analysis

#### Saliency Analysis (Part 2)
```bash
# my_cnn
cd my_cnn
python part2.py --model_path my_cnn_weights.pth --output_dir analysis_outputs
sbatch run_part2.slurm

# my_cnn2
cd my_cnn2
python part2_saliency.py --model_path my_cnn2_weights.pth
```

#### Research Analysis (Part 3)
```bash
# my_cnn
cd my_cnn
python part3.py --model_path my_cnn_weights.pth
sbatch run_part3.slurm

# my_cnn2
cd my_cnn2
python part3_analysis.py
```

## Key Outputs

### Trained Models
- `my_cnn/my_cnn_weights.pth` - Main weather CNN
- `my_cnn2/my_cnn2_weights.pth`, `my_cnn3_weights.pth` - Alternative models

### Metrics and Results
- `training_metrics.json` - Training curves, validation scores, accuracies
- Individual experiment metrics in respective output directories

### Analysis Outputs
- Saliency maps and visualizations
- Representation similarity matrices
- Transfer learning performance comparisons
- Research question analysis results

### Normalization Statistics
- Computed mean/std for data normalization
- Saved for reproducibility across experiments

## Notes

- Both implementations serve the same assignment requirements but with different code organization
- `my_cnn` has more comprehensive analysis tools and visualization
- `my_cnn2` includes additional model variants and testing utilities
- All training scripts support GPU acceleration and HPC job submission
- Pre-computed valid data indices avoid repeated data filtering
- Transfer learning experiments use standardized evaluation metrics (top-1, top-5 accuracy)

## Assignment Requirements

Refer to `assignment2.pdf` for detailed assignment specifications, grading criteria, and submission requirements.