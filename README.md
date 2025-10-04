# Deep Learning Project 2 - GPU Cluster Training Protocol

This document outlines the complete protocol for running the deep learning training pipeline on DTU's GPU cluster using LSF (Load Sharing Facility).

## Project Overview

This project implements and compares 6 different fusion models for video classification on the UFC-10 dataset:

- **2D Models**: PerFrameAggregation2D, LateFusion2D, EarlyFusion2D
- **3D Models**: PerFrameAggregation3D, LateFusion3D, EarlyFusion3D

## File Structure

```
Project2/
├── submit.sh              # LSF job submission script
├── train.py              # Main training script
├── datasets.py           # Dataset loading utilities
├── models.py             # Model definitions
├── networks.py           # Network architectures
├── requirements.txt      # Python dependencies
├── ufc10/                # Dataset directory
│   ├── frames/           # Extracted video frames
│   ├── videos/           # Original video files
│   └── metadata/         # CSV files for train/val/test splits
├── models/               # Saved model weights (created during training)
├── checkpoints/          # Training checkpoints (created during training)
└── logs/                 # Job output logs (created during training)
```

## Prerequisites

- Access to DTU's GPU cluster
- Basic familiarity with LSF job submission
- Python 3.9+ environment

## GPU Cluster Setup

### Available GPU Queues

Based on DTU's LSF10 setup, the following GPU queues are available:

| Queue Name | GPU Type | Memory | Nodes | Recommended For |
|------------|----------|--------|-------|------------------|
| `gpuv100`  | Tesla V100 16GB/32GB | 16-32GB | 14 nodes | **Recommended** - Good balance |
| `gpua100`  | Tesla A100 40GB/80GB | 40-80GB | 10 nodes | Large models, high memory needs |
| `gpul40s`  | L40s 48GB | 48GB | 6 nodes | High memory requirements |
| `gpua10`   | Tesla A10 24GB | 24GB | 1 node | Medium models |
| `gpua40`   | Tesla A40 48GB | 48GB | 1 node | High memory, NVLink |

### CUDA Requirements

- **A100 GPUs**: Require CUDA 11.0 or newer
- **Other GPUs**: CUDA 11.6 (as specified in script)

## Job Submission Protocol

### 1. Prepare Your Environment

Ensure all files are in the correct directory structure:
```bash
# Verify your project structure
ls -la
# Should show: submit.sh, train.py, datasets.py, models.py, networks.py, requirements.txt, ufc10/
```

### 2. Submit the Job

```bash
# Submit the job to the GPU cluster
bsub < submit.sh
```

### 3. Monitor Job Status

```bash
# Check job status
bjobs

# Check specific job details
bjobs -l <JOB_ID>

# Monitor real-time output
tail -f logs/training_<JOB_ID>.out
```

### 4. Job Management

```bash
# Cancel a job
bkill <JOB_ID>

# Check job history
bjobs -a

# View job output
cat logs/training_<JOB_ID>.out
cat logs/training_<JOB_ID>.err
```

## Script Configuration

### LSF Directives Explained

The `submit.sh` script contains the following LSF directives:

```bash
#BSUB -q gpuv100                    # Queue: Tesla V100 GPUs
#BSUB -J deep_learning_project2     # Job name
#BSUB -n 4                          # 4 CPU cores
#BSUB -gpu "num=1:mode=exclusive_process"  # 1 GPU, exclusive access
#BSUB -W 4:00                       # 4-hour time limit
#BSUB -R "rusage[mem=16GB]"         # 16GB system memory
#BSUB -B                            # Email notification on start
#BSUB -N                            # Email notification on completion
#BSUB -o logs/training_%J.out       # Output file
#BSUB -e logs/training_%J.err        # Error file
```

### Customization Options

#### Change GPU Queue
Edit line 3 in `submit.sh`:
```bash
#BSUB -q gpua100    # For A100 GPUs (more powerful)
#BSUB -q gpul40s    # For L40s GPUs (high memory)
```

#### Adjust Resources
```bash
#BSUB -W 8:00                       # 8-hour time limit
#BSUB -R "rusage[mem=32GB]"         # 32GB system memory
#BSUB -n 8                           # 8 CPU cores
```

#### Disable Email Notifications
Comment out these lines:
```bash
# #BSUB -B
# #BSUB -N
```

## Training Process

### What Happens During Training

1. **Environment Setup**: Python 3.9 and CUDA 11.6 modules are loaded
2. **Virtual Environment**: Created if it doesn't exist
3. **Dependencies**: Installed from `requirements.txt`
4. **Model Training**: All 6 models are trained sequentially:
   - PerFrameAggregation2D
   - LateFusion2D
   - EarlyFusion2D
   - PerFrameAggregation3D
   - LateFusion3D
   - EarlyFusion3D

### Output Files

After training, you'll find:

- **`models/`**: Best model weights for each architecture
- **`checkpoints/`**: Training checkpoints and history
- **`logs/`**: Job output and error logs

### Expected Training Time

- **Per model**: ~30-60 minutes (depending on GPU)
- **Total time**: ~3-6 hours for all 6 models
- **Recommended time limit**: 4-8 hours

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Increase memory allocation: `#BSUB -R "rusage[mem=32GB]"`
   - Use a GPU with more memory: `gpua100` or `gpul40s`

2. **Job Timeout**
   - Increase time limit: `#BSUB -W 8:00`
   - Monitor progress and adjust accordingly

3. **Module Loading Issues**
   - Check available modules: `module avail`
   - Verify CUDA version compatibility

4. **Dependency Installation Fails**
   - Check internet connectivity on cluster
   - Verify `requirements.txt` format

### Debugging Commands

```bash
# Check available modules
module avail

# Check GPU availability
nvidia-smi

# Test Python environment
python -c "import torch; print(torch.cuda.is_available())"

# Check job resource usage
bjobs -l <JOB_ID>
```

## Results Analysis

### Model Performance Comparison

After training completes, check the final results in the output log:
```
FINAL RESULTS SUMMARY
============================================================
PerFrame2D     :   XX.XX%
LateFusion2D   :   XX.XX%
EarlyFusion2D  :   XX.XX%
PerFrame3D     :   XX.XX%
LateFusion3D   :   XX.XX%
EarlyFusion3D  :   XX.XX%

Best performing model: <MODEL_NAME> (XX.XX%)
```

### Saved Files

- **Best weights**: `models/<MODEL_NAME>_best_weights.pth`
- **Final weights**: `models/<MODEL_NAME>_final_weights.pth`
- **Training history**: `checkpoints/<MODEL_NAME>_training_history.json`
- **Checkpoints**: `checkpoints/<MODEL_NAME>_epoch_*.pth`

## Best Practices

1. **Test Locally First**: Run a quick test on a small subset before submitting to cluster
2. **Monitor Resources**: Use `bjobs -l` to check resource usage
3. **Save Checkpoints**: The script automatically saves checkpoints for resuming
4. **Document Results**: Keep track of which models perform best
5. **Clean Up**: Remove old logs and checkpoints to save space

## Support

For technical issues with DTU's GPU cluster:
- Check DTU's HPC documentation
- Contact DTU Compute support
- Verify LSF queue availability with `bqueues`

For project-specific issues:
- Check the training logs in `logs/`
- Verify dataset structure in `ufc10/`
- Ensure all Python dependencies are correctly specified
