#!/bin/bash
set -e

if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment (.venv) not found. Please run ./setup.sh first."
    exit 1
fi
source .venv/bin/activate

if [ "$1" == "--web" ]; then
    echo "🚀 Starting ROCm-Pilot Web UI..."
    python3 -m app_web
else
    echo "🚀 Starting ROCm-Pilot CLI Agent..."
    python3 -m src.agent "$@"
fi
