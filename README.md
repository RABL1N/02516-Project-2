# Deep Learning Project 2 - Video Classification Pipeline

This project implements and compares 6 different fusion models for video classification on the UFC-10 dataset, with proper separation of training, testing, and analysis phases.

## Project Overview

This project implements and compares 6 different fusion models for video classification:

- **2D Models**: PerFrameAggregation2D, LateFusion2D, EarlyFusion2D
- **3D Models**: PerFrameAggregation3D, LateFusion3D, EarlyFusion3D

## Repository Structure

```
Project2/
├── train.py              # Main training script (trains all models)
├── eval.py               # Model evaluation script (tests trained models)
├── plotting.py           # Results analysis and visualization
├── submit.sh             # LSF job submission script for HPC
├── models.py             # Model architecture definitions
├── networks.py           # Core network components
├── datasets.py           # Dataset loading utilities
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── results.json          # Test results (created during evaluation)
├── ufc10/                # Dataset directory (not in repo)
│   ├── frames/           # Extracted video frames
│   ├── videos/           # Original video files
│   └── metadata/         # CSV files for train/val/test splits
├── models/               # Saved model weights (created during training)
├── logs/                 # Training histories (created during training)
├── runtime/              # Job output logs (created during training)
└── plots/                # Generated plots (created during analysis)
```

## Workflow Overview

The project follows a **3-phase workflow** with proper separation of concerns:

### Phase 1: Training (`train.py`)

- **Purpose**: Train all 6 models on the dataset
- **Output**: Model weights saved to `models/` directory
- **Best Practice**: Saves best weights based on validation accuracy
- **No test evaluation**: Test set remains truly unseen

### Phase 2: Evaluation (`eval.py`)

- **Purpose**: Load trained models and evaluate on test set
- **Output**: Test results saved to `results.json`
- **Best Practice**: Single evaluation of test set for final performance

### Phase 3: Analysis (`plotting.py`)

- **Purpose**: Generate visualizations and analysis plots
- **Output**: Training curves and test accuracy comparisons
- **Best Practice**: Uses actual test results from evaluation phase

## Script Usage

### 1. Training Phase

```bash
# Option A: Run locally (if you have GPU)
python train.py

# Option B: Submit to HPC cluster
bsub < submit.sh
```

**What happens:**

- Trains all 6 models sequentially
- Saves best weights to `models/` directory
- Saves training history to `logs/`
- **No test evaluation** - test set remains unseen

### 2. Evaluation Phase

```bash
# Test all trained models on test set
python eval.py
```

**What happens:**

- Loads best weights from `models/` directory
- Evaluates each model on test set
- Saves results to `results.json`
- **Single evaluation** - test set used only once

### 3. Analysis Phase

```bash
# Generate plots and analysis
python plotting.py
```

**What happens:**

- Loads training histories from `logs/`
- Loads test results from `results.json`
- Generates training curves and test accuracy plots
- Saves plots to `plots/` directory

## HPC Cluster Usage

### Job Submission

```bash
# Submit training job to GPU cluster
bsub < submit.sh
```

### Monitor Training

```bash
# Check job status
bjobs

# Monitor real-time output
tail -f runtime/training_<JOB_ID>.out

# Check for errors
tail -f runtime/training_<JOB_ID>.err
```

### After Training Completes

```bash
# Copy results back to local machine
scp -r s234806@login1.hpc.dtu.dk:~/02516/Project2/models ./
scp -r s234806@login1.hpc.dtu.dk:~/02516/Project2/logs ./
scp -r s234806@login1.hpc.dtu.dk:~/02516/Project2/runtime ./

# Run evaluation locally
python eval.py

# Generate analysis plots
python plotting.py
```

## Output Files

### Training Outputs

- **`models/<MODEL_NAME>_best_weights.pth`**: Best model weights (based on validation accuracy)
- **`models/<MODEL_NAME>_final_weights.pth`**: Final model weights (after all epochs)
- **`logs/<MODEL_NAME>_training_history.json`**: Complete training history

### Evaluation Outputs

- **`results.json`**: Test accuracy results for all models
- **Format**: `{"PerFrame2D": {"accuracy": 85.0, "loss": 0.595}, ...}`

### Analysis Outputs

- **`plots/training_curves.png`**: Training and validation curves for all models
- **`plots/testing_bars.png`**: Test accuracy comparison bar chart
