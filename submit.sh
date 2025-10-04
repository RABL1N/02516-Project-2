#!/bin/sh

#BSUB -q gpuv100

#BSUB -J deep_learning_project2

#BSUB -n 4

#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -W 4:00

#BSUB -R "rusage[mem=16GB]"

#BSUB -B

#BSUB -N

#BSUB -o logs/training_%J.out
#BSUB -e logs/training_%J.err

# Create logs directory if it doesn't exist
mkdir -p logs

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

# Run the training script
python train.py