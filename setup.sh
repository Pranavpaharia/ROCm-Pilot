#!/bin/bash
set -e

echo "============================================"
echo "  🚀 ROCm-Pilot Full Setup"
echo "============================================"

# Ensure sub-scripts are executable
chmod +x setup-deps.sh setup-docs.sh build-kb.sh run.sh

PROFILE="core"
if [ "$1" == "--full" ]; then
    PROFILE="full"
fi

echo "[1/3] Setting up dependencies..."
./setup-deps.sh

echo ""
echo "[2/3] Cloning documentation..."
./setup-docs.sh --$PROFILE

echo ""
echo "[3/3] Building Knowledge Base..."
./build-kb.sh

echo ""
echo "============================================"
echo "  ✅ ROCm-Pilot is ready!"
echo ""
echo "  Set your API key (if using cloud LLM):"
echo "    export FIREWORKS_API_KEY='your-key'"
echo ""
echo "  Start the CLI agent:"
echo "    ./run.sh"
echo ""
echo "  Or start the Web UI:"
echo "    ./run.sh --web"
echo "============================================"
