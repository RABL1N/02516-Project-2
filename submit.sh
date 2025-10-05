#!/bin/sh

#BSUB -q gpuv100

#BSUB -J deep_learning_project2

#BSUB -n 4

#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -W 4:00

#BSUB -R "rusage[mem=16GB]"

#BSUB -B

#BSUB -N

# Set dataset and dual-stream flag (can be changed for different experiments)
DATASET="ucf101_noleakage"
INCLUDE_DUAL_STREAM="--include-dual-stream"  # Set to "" to disable dual-stream models

# Determine results directory based on dual-stream flag
if [ -n "$INCLUDE_DUAL_STREAM" ]; then
    RESULTS_DIR="without_leakage_all"
else
    RESULTS_DIR="without_leakage"
fi

#BSUB -o ${RESULTS_DIR}/output/training_%J.out
#BSUB -e ${RESULTS_DIR}/output/training_%J.err

# Create output directory if it doesn't exist
mkdir -p ${RESULTS_DIR}/output
mkdir -p ${RESULTS_DIR}/models
mkdir -p ${RESULTS_DIR}/logs
mkdir -p ${RESULTS_DIR}/plots

# Load required modules
module load python3/3.9
module load cuda/11.6

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the complete workflow using main.py
echo "Starting complete workflow..."
echo "Dataset: ${DATASET}"
echo "Results directory: ${RESULTS_DIR}"
echo "Dual-stream models: ${INCLUDE_DUAL_STREAM}"
echo "=========================================="

python main.py --dataset ${DATASET} ${INCLUDE_DUAL_STREAM}