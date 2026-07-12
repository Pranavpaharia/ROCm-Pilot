#!/bin/bash
set -e
echo "============================================"
echo "  🧠 ROCm-Pilot ChromaDB Vector Build"
echo "============================================"

if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment (.venv) not found. Run ./setup-deps.sh first."
    exit 1
fi
source .venv/bin/activate

if [ -d "data/chroma_db" ] && [ "$(ls -A data/chroma_db 2>/dev/null)" ]; then
    echo "Knowledge base already exists — skipping build."
    echo "To rebuild, remove data/chroma_db and re-run this script."
    python3 -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma_db')
col = client.get_collection('rocm_docs')
print(f'  ✅ {col.count():,} chunks in vector store')
"
else
    echo "Building knowledge base (this may take a few minutes)..."
    python3 -c "
from src.scraper import collect_documents
from src.chunker import chunk_documents
from src.embedder import build_vector_store

docs = collect_documents('data/raw_docs')
chunks = chunk_documents(docs)
build_vector_store(chunks, 'data/chroma_db')
"
    echo "✅ Knowledge base built successfully."
fi
