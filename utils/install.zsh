#!/usr/bin/env zsh
set -e  # Stop execution if command fails

# Check architecture
ARCH=$(uname -m)

if [[ "$ARCH" == "arm64" ]]; then
    echo "AppleSilicon-based Mac detected."
    PYTHON_PATH=$(which python3.9)  # Usa direttamente Python ARM
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "Intel-based Mac detected."
    PYTHON_PATH=$(which python3.9)  # Usa direttamente Python Intel
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
        rm -rf libs/comet-emu
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

# Install the corrected comet-emu files
if [ ! -d "libs/comet-emu" ]; then
    echo "\n>>> Cloning comet-emu into libs/..."
    git clone git@gitlab.com:aegge/comet-emu.git libs/comet-emu
    
    echo ">>> Installing comet-emu..."
    pushd libs/comet-emu > /dev/null
    pip install -e .
    popd > /dev/null

    # Set up correct files
    EMU_DIR="utils/LPT_emulator"
    PTEMU_DIR="libs/comet-emu/comet"

    if [ -d "$PTEMU_DIR" ]; then
        cp "$EMU_DIR/tables.py" "$PTEMU_DIR/"
        cp "$EMU_DIR/PTEmu_LPT_new.py" "$PTEMU_DIR/"
        echo "  -> Replaced tables.py and added PTEmu_LPT_new.py into $PTEMU_DIR."
    else
        echo "  -> WARNING: PTEmu.py's folder not found. Could not route python scripts."
    fi

    # Place table_files and model_files into data_dir
    DATA_DIR="libs/comet-emu/comet/data_dir"

    if [ -d "$DATA_DIR" ]; then
        mkdir -p "$DATA_DIR/tables"
        mkdir -p "$DATA_DIR/models"

        # Copy the contents of the directories
        cp "$EMU_DIR/table_files"/* "$DATA_DIR/tables/"
        echo "  -> Copied table files."

        cp "$EMU_DIR/model_files"/* "$DATA_DIR/models/"
        echo "  -> Copied model files."
    else
        echo "  -> WARNING: data_dir not found in comet-emu. Skipping data files."
    fi

    echo ">>> comet-emu patch applied successfully!"

else
    echo "\n>>> comet-emu is already cloned in libs/"
fi

# Add the libs directory to the virtual environment
pip install -e .

echo "\nVirtual environment setup complete!"