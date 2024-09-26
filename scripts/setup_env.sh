#!/bin/bash

# Script to set up h2o conda environment and jupyter kernel

# Use the environment yaml file to create the environment
echo "Creating conda environment from the yaml file..."
conda env create -f ../config/h2o_env.yml

# Activate environment
echo "Activating the environment..."
conda activate payments_env

# Create Jupyter kernel with environment. Analysis developed interactively in VS code, but doesn't have to be used when reproduced
conda install -c conda-forge jupyter ipykernel -y

echo "Creating jupyter kernel..."
python -m ipykernel install --user --name payments_env --display-name "Python (payments_env)"

# Make script executable
chmod +x setup_env.sh

echo "Set-up complete! Environment payments_env is ready to use"
