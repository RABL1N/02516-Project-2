#!/bin/sh

#BSUB -q gpuv100

#BSUB -J deep_learning_project2

#BSUB -n 4

#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -W 4:00

#BSUB -R "rusage[mem=16GB]"

#BSUB -B

#BSUB -N

# Set dataset and results directory (can be changed for different experiments)
DATASET="ucf101_noleakage"
RESULTS_DIR="without_leakage"

#BSUB -o ${RESULTS_DIR}/output/training_%J.out
#BSUB -e ${RESULTS_DIR}/output/training_%J.err

# Create output directory if it doesn't exist
mkdir -p ${RESULTS_DIR}/output

# Load required modules
module load python/3.9
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

# Run the complete workflow
python train.py ${DATASET} ${RESULTS_DIR}
python eval.py ${DATASET} ${RESULTS_DIR}
python plotting.py ${RESULTS_DIR}