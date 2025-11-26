#!/bin/bash

# Script to create a Python virtual environment and install vamp-planner

# Default venv name
VENV_NAME="venv"

# Allow custom venv name as first argument
if [ -n "$1" ]; then
    VENV_NAME="$1"
fi

# Check if venv already exists
if [ -d "$VENV_NAME" ]; then
    echo "Error: Virtual environment '$VENV_NAME' already exists."
    echo "Please remove it first or choose a different name."
    exit 1
fi

# Create the virtual environment
echo "Creating virtual environment: $VENV_NAME"
python3 -m venv "$VENV_NAME"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi

echo "Virtual environment '$VENV_NAME' created successfully!"
echo ""

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install build dependencies first (required by scikit-build-core)
echo "Installing build dependencies..."
pip install "scikit-build-core>=0.4.3" "nanobind>=2.0.0" "cmake>=3.1" "typing_extensions"

if [ $? -ne 0 ]; then
    echo "Error: Failed to install build dependencies."
    exit 1
fi

# Install the package in editable mode with examples
echo "Installing vamp-planner in editable mode with examples..."
pip install --no-build-isolation -Ceditable.rebuild=true -ve .[examples]

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Setup completed successfully!"
    echo "Virtual environment is now activated."
    echo "======================================"
else
    echo ""
    echo "Error: Failed to install vamp-planner."
    echo "You may need to install system dependencies (e.g., cmake, build tools)."
    exit 1
fi