#!/bin/bash
set -e
echo "============================================"
echo "  📚 ROCm-Pilot Knowledge Base Setup"
echo "============================================"

PROFILE="core"
if [ "$1" == "--full" ]; then
    PROFILE="full"
fi

echo "  ↓ Cloning AMD documentation (Profile: $PROFILE)..."
mkdir -p data/raw_docs

clone_if_missing() {
    local repo_url=$1
    local target_dir=$2
    if [ ! -d "$target_dir" ]; then
        echo "  ↓ Cloning $(basename $target_dir)..."
        git clone --depth 1 --quiet "$repo_url" "$target_dir"
    else
        echo "  ✓ $(basename $target_dir) already exists"
    fi
}

# Core Docs (Always cloned)
clone_if_missing "https://github.com/ROCm/ROCm.git" "data/raw_docs/ROCm"
clone_if_missing "https://github.com/ROCm/rocm-install-on-linux.git" "data/raw_docs/rocm-install-on-linux"

if [ "$PROFILE" == "full" ]; then
    echo "  ↓ Cloning extended documentation repositories..."
    clone_if_missing "https://github.com/ROCm/HIP.git" "data/raw_docs/HIP"
    clone_if_missing "https://github.com/ROCm/MIOpen.git" "data/raw_docs/MIOpen"
    clone_if_missing "https://github.com/ROCm/AMDMIGraphX.git" "data/raw_docs/AMDMIGraphX"
    clone_if_missing "https://github.com/ROCm/rocm-blogs.git" "data/raw_docs/rocm-blogs"
    clone_if_missing "https://github.com/ROCm/gpuaidev.git" "data/raw_docs/gpuaidev"
    clone_if_missing "https://github.com/amd/RyzenAI-SW.git" "data/raw_docs/ryzen-ai-sw"
    clone_if_missing "https://github.com/lemonade-sdk/lemonade.git" "data/raw_docs/lemonade"
else
    echo "  ℹ️  Skipping extended repos (HIP, MIOpen, RyzenAI-SW, etc.). Run with --full to include."
fi

# Fetch latest GPU compatibility data
echo ""
echo "  ↓ Fetching latest GPU compatibility data..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    python3 -c "
from src.live_scraper import AMDDocsScraper
scraper = AMDDocsScraper()
result = scraper.update_gpu_database('data/gpu_database.json')
gpu_count = len(result.get('gpu_architectures', {}))
print(f'  ✅ Updated {gpu_count} GPU architecture entries')
" 2>/dev/null || echo "  ⚠️  Live scraping failed, using bundled GPU database"
else
    echo "  ⚠️  Python environment not found (.venv missing). Please run setup-deps.sh first."
fi

echo "✅ Documentation repos ready"
