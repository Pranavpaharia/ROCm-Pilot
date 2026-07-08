"""
Retriever: queries the ChromaDB vector store for relevant documentation chunks.
"""

from typing import List, Dict, Optional


def get_retriever(
    db_path: str = 'data/chroma_db',
    collection_name: str = 'rocm_docs',
):
    """
    Get a handle to the ChromaDB collection for querying.

    Args:
        db_path: Path to persistent ChromaDB storage.
        collection_name: Name of the collection to query.

    Returns:
        A ChromaDB Collection object.
    """
    import chromadb

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    return collection


def retrieve(
    query: str,
    collection,
    embedding_model=None,
    top_k: int = 8,
    doc_type_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Retrieve the most relevant documentation chunks for a user query.

    Args:
        query: The user's natural-language question.
        collection: A ChromaDB collection handle.
        embedding_model: Optional SentenceTransformer model for encoding
                         the query. If None, ChromaDB's default is used.
        top_k: Number of results to return.
        doc_type_filter: Optional filter on document type
                         (e.g., "installation", "blog", "tutorial").

    Returns:
        List of result dicts with keys: text, metadata, distance, id.
    """
    where_filter = None
    if doc_type_filter:
        where_filter = {"doc_type": doc_type_filter}

    if embedding_model is not None:
        query_embedding = embedding_model.encode(
            [query], convert_to_numpy=True
        ).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    else:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

    # Flatten ChromaDB's nested list structure
    formatted = []
    for i in range(len(results['ids'][0])):
        formatted.append({
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i],
            'id': results['ids'][0][i],
        })

    return formatted


def format_context(results: List[Dict], max_words: int = 3000) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.

    Includes source citations so the LLM can reference them in its answer.

    Args:
        results: List of result dicts from retrieve().
        max_words: Approximate word-count cap for the combined context.

    Returns:
        A formatted string with numbered source blocks.
    """
    context_parts = []
    total_words = 0

    for i, result in enumerate(results, 1):
        meta = result['metadata']
        text = result['text']

        words = len(text.split())
        if total_words + words > max_words:
            break

        header = (
            f"[Source {i}: {meta['source_repo']}/{meta['source_file']} "
            f"| Type: {meta['doc_type']} "
            f"| Section: {meta['section_title']}]"
        )
        context_parts.append(f"{header}\n{text}")
        total_words += words

    return "\n\n---\n\n".join(context_parts)
