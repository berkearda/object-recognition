#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Miniconda or Anaconda."
    exit 1
fi

# Create Conda environment if it doesn't exist
if ! conda info --envs | grep ex2; then
    echo "Creating Conda environment 'ex2' with Python 3.10..."
    conda create -n ex2 python=3.10 -y
else
    echo "Environment 'ex2' already exists."
fi

# Activate the Conda environment
echo "Activating Conda environment 'ex2'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ex2

# Install dependencies using pip (you can also use conda install if necessary)
echo "Installing dependencies..."
pip install scikit-learn opencv-python matplotlib tqdm

echo "All dependencies installed successfully!"

# Deactivate the Conda environment
echo "Deactivating Conda environment..."
conda deactivate

# Provide instructions to activate the environment
echo "Setup complete! To activate the environment, run: 'conda activate ex2'"
