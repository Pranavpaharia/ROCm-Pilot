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

# 1. Install ROCm-compatible PyTorch if not already available
PYTHON_BIN="${PYTHON_CMD:-python3}"
echo "[1/4] Checking for PyTorch..."
if "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
    echo "  ✓ PyTorch is already installed. Skipping installation..."
else
    IS_DARWIN=$(uname -s | grep -i -q "Darwin" && echo "true" || echo "false")
    if [ "$IS_DARWIN" = "true" ]; then
        echo "  ↓ Installing standard PyTorch for macOS..."
        "$PYTHON_BIN" -m pip install --break-system-packages torch torchvision torchaudio
        echo "✅ PyTorch (Mac) installed"
    else
        echo "  ↓ Purging any apt-installed PyTorch packages to prevent conflicts..."
        apt-get remove -y python3-torch python3-torchvision python3-torchaudio python3-typing-extensions 2>/dev/null || true
        echo "  ↓ Uninstalling any existing pip PyTorch versions..."
        "$PYTHON_BIN" -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        echo "  ↓ Installing PyTorch + torchvision + torchaudio (ROCm 6.2 support)..."
        "$PYTHON_BIN" -m pip install --break-system-packages --ignore-installed \
            torch==2.5.1+rocm6.2 \
            torchvision==0.20.1+rocm6.2 \
            torchaudio==2.5.1+rocm6.2 \
            --index-url https://download.pytorch.org/whl/rocm6.2
        echo "✅ PyTorch ROCm installed"
    fi
fi
echo ""

echo "  ↓ Purging potential conflicting packages..."
"$PYTHON_BIN" -m pip uninstall -y transformers sentence-transformers chromadb 2>/dev/null || true

echo "  ↓ Installing additional dependencies..."
"$PYTHON_BIN" -m pip install --break-system-packages \
    chromadb \
    sentence-transformers \
    transformers==4.45.2 \
    openai tqdm ipywidgets
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
"$PYTHON_BIN" -c "
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
    "$PYTHON_BIN" -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma_db')
col = client.get_collection('rocm_docs')
print(f'  ✅ {col.count():,} chunks in vector store')
"
else
    echo "[4/4] Building knowledge base (first run — may take a few minutes)..."
    "$PYTHON_BIN" -c "
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
