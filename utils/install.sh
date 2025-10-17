#!/bin/bash

# Check if the venv folder already exists
if [ -d "venv" ]; then
    echo "The 'venv' folder already exists."
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo # move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove the existing folder
        rm -rf venv
        rm -rf lab_astro_libs.egg-info
    else
        echo "Operation cancelled."
        exit 1
    fi
fi

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

pip install --upgrade pip

# Install packages
pip install -r utils/requirements.txt

# Add the libs directory to the virtual environment
pip install -e .
