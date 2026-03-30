#!/bin/bash
set -e 

PYTHON_EXE=$(which python3)

# Check if the venv folder already exists
if [ -d "venv" ]; then
    echo "The 'venv' folder already exists."
    read -p "Do you want to remove it and create a new one? (y/n) " reply
    if [[ $reply =~ ^[Yy]$ ]]; then
        rm -rf venv
        rm -rf lab_astro_libs.egg-info
        rm -rf libs/comet-emu
    else
        echo "Operation cancelled."
        exit 1
    fi
fi

# Create a virtual environment with the correct interpreter
$PYTHON_EXE -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
if [ -f "utils/requirements.txt" ]; then
    pip install -r utils/requirements.txt
fi

# Install the corrected comet-emu files
if [ ! -d "libs/comet-emu" ]; then
    echo -e "\n>>> Cloning comet-emu into libs/..."
    git clone https://gitlab.com/aegge/comet-emu.git libs/comet-emu
    
    echo ">>> Installing comet-emu..."
    cd libs/comet-emu
    pip install -e .
    cd ../..

    # Place table_files and model_files into data_dir
    EMU_DIR="utils/LPT_emulator"
    PTEMU_DIR="libs/comet-emu/comet"

    if [ -d "$PTEMU_DIR" ]; then
        cp "$EMU_DIR/tables.py" "$PTEMU_DIR/"
        cp "$EMU_DIR/PTEmu_LPT_new.py" "$PTEMU_DIR/"
        
        DATA_DIR="libs/comet-emu/comet/data_dir"
        mkdir -p "$DATA_DIR/tables" "$DATA_DIR/models"
        cp "$EMU_DIR/table_files"/* "$DATA_DIR/tables/" 2>/dev/null || true
        cp "$EMU_DIR/model_files"/* "$DATA_DIR/models/" 2>/dev/null || true
        echo ">>> comet-emu patch applied."
    fi
fi

# Add the libs directory to the virtual environment
pip install -e .

echo "\nVirtual environment setup complete!"