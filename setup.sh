#!/bin/bash
# ============================================
#  ROCm-Pilot Setup Script
#  Run this on the AMD Jupyter instance
# ============================================

set -e

echo "============================================"
echo "  🚀 ROCm-Pilot Setup Script"
echo "============================================"
echo ""

# 1. Install PyTorch with ROCm support
echo "[1/5] Installing PyTorch with ROCm support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
echo "✅ PyTorch installed"
echo ""

# 2. Install Python dependencies
echo "[2/5] Installing Python dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# 3. Clone documentation repos (shallow clones for speed)
echo "[3/5] Cloning AMD ROCm documentation repos..."
mkdir -p data/raw_docs

if [ ! -d "data/raw_docs/ROCm" ]; then
    echo "  Cloning ROCm/ROCm..."
    git clone --depth 1 https://github.com/ROCm/ROCm.git data/raw_docs/ROCm
else
    echo "  ROCm/ROCm already exists, skipping."
fi

if [ ! -d "data/raw_docs/rocm-install-on-linux" ]; then
    echo "  Cloning ROCm/rocm-install-on-linux..."
    git clone --depth 1 https://github.com/ROCm/rocm-install-on-linux.git data/raw_docs/rocm-install-on-linux
else
    echo "  ROCm/rocm-install-on-linux already exists, skipping."
fi

if [ ! -d "data/raw_docs/rocm-blogs" ]; then
    echo "  Cloning ROCm/rocm-blogs..."
    git clone --depth 1 https://github.com/ROCm/rocm-blogs.git data/raw_docs/rocm-blogs
else
    echo "  ROCm/rocm-blogs already exists, skipping."
fi

if [ ! -d "data/raw_docs/gpuaidev" ]; then
    echo "  Cloning ROCm/gpuaidev..."
    git clone --depth 1 https://github.com/ROCm/gpuaidev.git data/raw_docs/gpuaidev
else
    echo "  ROCm/gpuaidev already exists, skipping."
fi

echo "✅ Documentation repos cloned"
echo ""

# 4. Verify GPU
echo "[4/5] Verifying AMD GPU..."
rocm-smi 2>/dev/null || echo "⚠️  rocm-smi not found (GPU detection may be limited)"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm (HIP) available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_mem / (1024**3):.1f} GB')
else:
    print('No GPU detected — embeddings will run on CPU (slower but functional)')
"
echo ""

# 5. Build knowledge base
echo "[5/5] Building knowledge base (this may take a few minutes)..."
python3 -c "
from src.scraper import collect_documents
from src.chunker import chunk_documents
from src.embedder import build_vector_store

print('Step 1: Collecting documents...')
docs = collect_documents('data/raw_docs')
print(f'  Found {len(docs)} documents')

print('Step 2: Chunking documents...')
chunks = chunk_documents(docs)
print(f'  Created {len(chunks)} chunks')

print('Step 3: Embedding & storing (GPU-accelerated)...')
build_vector_store(chunks, 'data/chroma_db')
print('  Done!')
"
echo ""

echo "============================================"
echo "  ✅ ROCm-Pilot setup complete!"
echo ""
echo "  To start the agent:"
echo "    export FIREWORKS_API_KEY='your-key-here'"
echo "    python3 -m src.agent"
echo "============================================"
