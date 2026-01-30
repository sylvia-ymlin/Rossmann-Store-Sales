#!/bin/bash

# Project Name for Conda Environment
ENV_NAME="stores-prediction"

echo "Creating Conda environment: $ENV_NAME..."

# Create environment with Python 3.12
conda create -y -n $ENV_NAME python=3.12

# Initialize conda for the current shell session
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Installing project dependencies..."
pip install -r requirements.txt

echo "Registering environment as Jupyter kernel..."
python -m ipykernel install --user --name $ENV_NAME --display-name "Python (Stores Prediction)"

echo "------------------------------------------------"
echo "Setup complete!"
echo "To activate the environment: conda activate $ENV_NAME"
echo "------------------------------------------------"
