"""
Embedding generator and ChromaDB vector store builder.
Uses sentence-transformers with AMD GPU acceleration via ROCm.
"""

import os
import torch
from typing import List, Dict
from tqdm import tqdm

# Import config for consistent configuration handling
from src.config import config


# Default settings for Embedder (supports Lemonade SDK and Fireworks AI)
DEFAULT_MODEL = config.DEFAULT_EMBEDDING_MODEL
BATCH_SIZE = 64


def get_api_config() -> tuple:
    """Get the correct API base URL and Key based on environment toggle."""
    use_local = os.environ.get('USE_LOCAL_LEMONADE', 'true').lower() == 'true'
    if use_local:
        print("🎯 Using Local SentenceTransformer for Embeddings")
        return 'local', 'not-needed'
    else:
        print("☁️ Using Fireworks API for Embeddings")
        api_key = os.environ.get('FIREWORKS_API_KEY')
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable is required when USE_LOCAL_LEMONADE=false")
        return 'https://api.fireworks.ai/inference/v1', api_key


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
    import requests

    api_base, api_key = get_api_config()
    print(f"Using embedding model: {model_name} at {api_base}")

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
            'hardware_target': c.get('hardware_target', 'General AMD'),
            'category': c.get('category', 'General'),
            'section_title': c['section_title'],
            'chunk_index': c['chunk_index'],
        }
        for c in chunks
    ]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Embed in batches
    all_embeddings = []
    
    if api_base == 'local':
        from sentence_transformers import SentenceTransformer
        print(f"Loading local model {model_name}...")
        model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            model = model.to('cuda')
        print(f"Embedding {len(texts)} chunks locally (batch_size={BATCH_SIZE})...")
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🧠 Embedding", unit="batch"):
            batch = texts[i : i + BATCH_SIZE]
            embeddings = model.encode(batch, convert_to_numpy=True).tolist()
            all_embeddings.extend(embeddings)
    else:
        print(f"Embedding {len(texts)} chunks via API (batch_size={BATCH_SIZE})...")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🧠 Embedding", unit="batch"):
            batch = texts[i : i + BATCH_SIZE]
            import requests
            response = requests.post(
                f"{api_base}/embeddings",
                headers=headers,
                json={"model": model_name, "input": batch}
            )
            response.raise_for_status()
            data = response.json()
            batch_embeddings = [item['embedding'] for item in sorted(data['data'], key=lambda x: x['index'])]
            all_embeddings.extend(batch_embeddings)

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
