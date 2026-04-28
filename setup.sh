#!/bin/bash

echo "Creating virtual environment..."

python3 -m venv .venv

echo "Activating virtual environment..."

source .venv/bin/activate

echo "Upgrading pip..."

pip install --upgrade pip

echo "Installing dependencies..."

pip install numpy pandas matplotlib seaborn scikit-learn scipy imbalanced-learn torch torchvision tqdm jupyter

echo "Freezing dependencies..."

pip freeze > requirements.txt

echo "Setup complete!"
echo "To activate later, run:"
echo "source .venv/bin/activate"