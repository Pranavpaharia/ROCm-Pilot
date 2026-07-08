#!/bin/bash
# ============================================
#  ROCm-Pilot Setup Script
#  Optimized for notebooks.amd.com
#  (PyTorch 2.9 + ROCm 7.2 are PRE-INSTALLED)
# ============================================

set -e

echo "============================================"
echo "  🚀 ROCm-Pilot Setup Script"
echo "  (Optimized — skips pre-installed packages)"
echo "============================================"
echo ""

# 1. Force clean reinstall of PyTorch with ROCm support
echo "[1/4] Installing ROCm-compatible PyTorch..."
echo "  ↓ Purging any apt-installed PyTorch packages to prevent conflicts..."
apt-get remove -y python3-torch python3-torchvision python3-torchaudio python3-typing-extensions 2>/dev/null || true
echo "  ↓ Uninstalling any existing pip PyTorch versions..."
pip uninstall -y --break-system-packages torch torchvision torchaudio || true
echo "  ↓ Installing PyTorch + torchvision + torchaudio (ROCm support)..."
pip install --break-system-packages --ignore-installed torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
echo "✅ PyTorch ROCm installed"
echo ""

echo "  ↓ Installing additional dependencies..."
pip install --quiet --break-system-packages --ignore-installed chromadb sentence-transformers openai tqdm ipywidgets
echo "✅ Dependencies installed"
echo ""

# 2. Clone documentation repos (shallow clones — fast)
echo "[2/4] Cloning AMD ROCm documentation repos..."
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

clone_if_missing "https://github.com/ROCm/ROCm.git" "data/raw_docs/ROCm"
clone_if_missing "https://github.com/ROCm/rocm-install-on-linux.git" "data/raw_docs/rocm-install-on-linux"
clone_if_missing "https://github.com/ROCm/rocm-blogs.git" "data/raw_docs/rocm-blogs"
clone_if_missing "https://github.com/ROCm/gpuaidev.git" "data/raw_docs/gpuaidev"

echo "✅ Documentation repos ready"
echo ""

# 3. Verify GPU
echo "[3/4] Verifying AMD GPU..."
rocm-smi 2>/dev/null || echo "⚠️  rocm-smi not available"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  ROCm/HIP available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  VRAM: {props.total_mem / (1024**3):.1f} GB')
" 2>/dev/null || echo "⚠️  PyTorch GPU check failed"
echo ""

# 4. Build knowledge base (skip if already exists)
if [ -d "data/chroma_db" ] && [ "$(ls -A data/chroma_db 2>/dev/null)" ]; then
    echo "[4/4] Knowledge base already exists — skipping build"
    python3 -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma_db')
col = client.get_collection('rocm_docs')
print(f'  ✅ {col.count():,} chunks in vector store')
"
else
    echo "[4/4] Building knowledge base (first run — may take a few minutes)..."
    python3 -c "
from src.scraper import collect_documents
from src.chunker import chunk_documents
from src.embedder import build_vector_store

docs = collect_documents('data/raw_docs')
chunks = chunk_documents(docs)
build_vector_store(chunks, 'data/chroma_db')
"
fi

echo ""
echo "============================================"
echo "  ✅ ROCm-Pilot is ready!"
echo ""
echo "  Set your API key:"
echo "    export FIREWORKS_API_KEY='your-key'"
echo ""
echo "  Start the agent:"
echo "    python3 -m src.agent"
echo "============================================"
