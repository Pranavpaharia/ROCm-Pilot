"""
Retriever: queries the ChromaDB vector store for relevant documentation chunks.
"""

import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger("rocm_pilot.retriever")


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    GPU-accelerated Cross-Encoder reranker using ms-marco-MiniLM-L-6-v2.

    The model is loaded lazily on first call to :meth:`rerank` and placed on
    the AMD GPU (``device='cuda'``) when available, falling back to CPU.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        self._model = None
        self._device: Optional[str] = None

    # -- lazy loading -------------------------------------------------------
    def _load_model(self):
        import torch
        from sentence_transformers import CrossEncoder

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading Cross-Encoder model '%s' on device '%s'",
            self.MODEL_NAME,
            self._device,
        )
        self._model = CrossEncoder(self.MODEL_NAME, device=self._device)
        logger.info("Cross-Encoder model loaded successfully on '%s'", self._device)

    # -- public API ---------------------------------------------------------
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 8,
    ) -> List[Dict]:
        """
        Score each result against *query* and return the *top_k* highest.

        Args:
            query: The user's natural-language question.
            results: List of result dicts (must contain a ``'text'`` key).
            top_k: Number of top-scoring results to return.

        Returns:
            Re-ranked list of result dicts, best first.
        """
        if not results:
            return results

        if self._model is None:
            self._load_model()

        # Build (query, passage) pairs for the Cross-Encoder
        pairs = [[query, r["text"]] for r in results]

        start = time.perf_counter()
        scores = self._model.predict(pairs)
        elapsed = time.perf_counter() - start

        logger.info(
            "Cross-Encoder reranked %d candidates in %.3f s (device=%s)",
            len(pairs),
            elapsed,
            self._device,
        )

        # Attach scores and sort descending
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        ranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Lazy singleton accessor
# ---------------------------------------------------------------------------

_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Return the module-level :class:`CrossEncoderReranker` singleton."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


# ---------------------------------------------------------------------------
# ChromaDB collection handle
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Retrieval (with optional Cross-Encoder reranking)
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    collection,
    embedding_model=None,
    top_k: int = 8,
    doc_type_filter: Optional[str] = None,
    use_reranker: bool = True,
) -> List[Dict]:
    """
    Retrieve the most relevant documentation chunks for a user query.

    When *use_reranker* is ``True`` (the default), the function over-fetches
    ``top_k * 3`` candidates from ChromaDB and then re-scores them with a
    Cross-Encoder model to return the best *top_k* results.

    Args:
        query: The user's natural-language question.
        collection: A ChromaDB collection handle.
        embedding_model: Optional SentenceTransformer model for encoding
                         the query. If None, ChromaDB's default is used.
        top_k: Number of results to return.
        doc_type_filter: Optional filter on document type
                         (e.g., "installation", "blog", "tutorial").
        use_reranker: If True, apply Cross-Encoder reranking. Defaults to True.

    Returns:
        List of result dicts with keys: text, metadata, distance, id.
    """
    # Determine how many candidates to fetch from ChromaDB
    fetch_k = top_k * 3 if use_reranker else top_k

    where_filter = None
    if doc_type_filter:
        where_filter = {"doc_type": doc_type_filter}

    chroma_start = time.perf_counter()

    if embedding_model is not None:
        query_embedding = embedding_model.encode(
            [query], convert_to_numpy=True
        ).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    else:
        results = collection.query(
            query_texts=[query],
            n_results=fetch_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

    chroma_elapsed = time.perf_counter() - chroma_start
    logger.info(
        "ChromaDB retrieval returned %d candidates in %.3f s",
        len(results['ids'][0]),
        chroma_elapsed,
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

    # Optional Cross-Encoder reranking
    if use_reranker and formatted:
        reranker = get_reranker()
        formatted = reranker.rerank(query, formatted, top_k=top_k)

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
            f"| Section: {meta['section_title']} "
            f"| URL: {meta.get('source_url', 'N/A')}]"
        )
        context_parts.append(f"{header}\n{text}")
        total_words += words

    return "\n\n---\n\n".join(context_parts)
