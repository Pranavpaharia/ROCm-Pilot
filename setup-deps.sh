#!/bin/bash
set -e
echo "============================================"
echo "  📦 ROCm-Pilot Dependency Setup (uv)"
echo "============================================"

# Install uv if missing
if ! command -v uv &> /dev/null; then
    echo "  ↓ Installing 'uv' package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Ensure uv is in PATH if installed locally
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

if [ ! -d ".venv" ]; then
    echo "  ↓ Creating virtual environment (.venv)..."
    uv venv .venv
else
    echo "  ✓ Virtual environment (.venv) already exists."
fi

echo "  ↓ Activating virtual environment..."
source .venv/bin/activate

echo "  ↓ Installing core dependencies..."
uv pip install -r requirements.txt

# Detect OS/Hardware for PyTorch
IS_DARWIN=$(uname -s | grep -i -q "Darwin" && echo "true" || echo "false")

echo "  ↓ Detecting hardware for PyTorch installation..."
if [ "$IS_DARWIN" = "true" ]; then
    echo "  → macOS detected. Installing CPU PyTorch..."
    uv pip install -r requirements-cpu.txt
else
    if command -v rocm-smi &> /dev/null; then
        echo "  → AMD GPU (ROCm) detected. Installing ROCm PyTorch..."
        uv pip install -r requirements-rocm.txt
    else
        echo "  → No ROCm detected. Defaulting to CPU PyTorch..."
        uv pip install -r requirements-cpu.txt
    fi
fi

echo "✅ Dependencies successfully installed in .venv"
