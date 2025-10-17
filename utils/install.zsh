#!/usr/bin/env zsh
set -e  # Stop execution if command fails

# Check architecture
ARCH=$(uname -m)

if [[ "$ARCH" == "arm64" ]]; then
    echo "AppleSilicon-based Mac detected."
    PYTHON_PATH=$(which python3)  # Usa direttamente Python ARM
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "Intel-based Mac detected."
    PYTHON_PATH=$(which python3)  # Usa direttamente Python Intel
else
    echo "Unknown architecture: $ARCH"
    exit 1
fi

# Check if the venv folder already exists
if [ -d "venv" ]; then
    echo "The 'venv' folder already exists."
    read "reply?Do you want to remove it and create a new one? (y/n) "
    echo    # (optional) move to a new line
    if [[ $reply =~ ^[Yy]$ ]]; then
        rm -rf venv
        rm -rf lab_astro_libs.egg-info
    else
        echo "Operation cancelled."
        exit 1
    fi
fi

# Create a virtual environment with the correct interpreter
$PYTHON_PATH -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install packages
pip install -r utils/requirements.txt

# Add the libs directory to the virtual environment
pip install -e .

echo "\nVirtual environment setup complete!"