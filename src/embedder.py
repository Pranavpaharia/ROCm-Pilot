"""
Embedding generator and ChromaDB vector store builder.
Uses sentence-transformers with AMD GPU acceleration via ROCm.
"""

import os
import torch
from typing import List, Dict
from tqdm import tqdm


# Embedding model — small, fast, and high quality
DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 64


def get_device() -> str:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"🎯 Using AMD GPU: {device_name}")
        return 'cuda'
    else:
        print("⚠️  No GPU detected, falling back to CPU (embeddings will be slower)")
        return 'cpu'


def build_vector_store(
    chunks: List[Dict],
    db_path: str = 'data/chroma_db',
    model_name: str = DEFAULT_MODEL,
    collection_name: str = 'rocm_docs',
) -> None:
    """
    Embed all document chunks and store them in a ChromaDB collection.

    Args:
        chunks: List of chunk dicts from chunker.chunk_documents().
        db_path: Filesystem path for ChromaDB persistent storage.
        model_name: HuggingFace model name for sentence embeddings.
        collection_name: Name of the ChromaDB collection.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    device = get_device()

    # Load the embedding model onto the detected device
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Initialize persistent ChromaDB
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    # Drop the collection if it already exists (ensures a clean rebuild)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare data
    texts = [c['text'] for c in chunks]
    metadatas = [
        {
            'source_repo': c['source_repo'],
            'source_file': c['source_file'],
            'source_url': c['source_url'],
            'doc_type': c['doc_type'],
            'section_title': c['section_title'],
            'chunk_index': c['chunk_index'],
        }
        for c in chunks
    ]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Embed in batches (GPU-accelerated)
    print(f"Embedding {len(texts)} chunks (batch_size={BATCH_SIZE})...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🧠 Embedding", unit="batch"):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.extend(embeddings.tolist())

    # Insert into ChromaDB in batches (ChromaDB caps at ~5000 per call)
    CHROMA_BATCH = 5000
    for i in tqdm(range(0, len(texts), CHROMA_BATCH), desc="💾 Storing", unit="batch"):
        end = min(i + CHROMA_BATCH, len(texts))
        collection.add(
            ids=ids[i:end],
            embeddings=all_embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"✅ Vector store built: {collection.count()} chunks stored in {db_path}")


if __name__ == '__main__':
    from src.scraper import collect_documents
    from src.chunker import chunk_documents

    docs = collect_documents('data/raw_docs')
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
