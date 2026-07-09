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
    hardware_target_filter: Optional[str] = None,
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

    where_filters = []
    if doc_type_filter:
        where_filters.append({"doc_type": doc_type_filter})
    if hardware_target_filter:
        where_filters.append({"hardware_target": hardware_target_filter})
        
    if len(where_filters) == 1:
        where_filter = where_filters[0]
    elif len(where_filters) > 1:
        where_filter = {"$and": where_filters}
    else:
        where_filter = None

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


# ---------------------------------------------------------------------------
# Query Classification & Smart Routing
# ---------------------------------------------------------------------------

# Patterns that indicate a structured GPU/version lookup
_GPU_PATTERNS = [
    r'\bgfx\d{3,4}\b',                          # gfx906, gfx1100, etc.
    r'\b(?:MI\s*\d{2,3}[AX]?)\b',               # MI100, MI300X, MI250, MI300A
    r'\b(?:RX\s*\d{4}\s*(?:XT|XTX)?)\b',        # RX 7900 XTX, RX 6800 XT
    r'\bRadeon\s+(?:VII|Pro|PRO)\b',             # Radeon VII, Radeon Pro
    r'\b(?:Instinct)\b',                          # AMD Instinct
    r'\b(?:CDNA|RDNA)\d?\b',                      # CDNA, CDNA2, RDNA3
]

_VERSION_PATTERNS = [
    r'\bROCm\s*[\d.]+\b',                         # ROCm 6.2, ROCm 7.1
    r'\brocm\s*[\d.]+\b',                         # rocm6.2
    r'\bPyTorch\s*[\d.]+\b',                      # PyTorch 2.5
    r'\btorch\s*[\d.]+\b',                         # torch 2.5
]

_STRUCTURED_KEYWORDS = {
    'compatible', 'compatibility', 'support', 'supported',
    'version', 'versions', 'which rocm', 'what rocm',
    'which pytorch', 'what pytorch', 'pip install',
    'install command', 'docker image',
}

_SEMANTIC_KEYWORDS = {
    'how to', 'how do', 'tutorial', 'guide', 'explain',
    'configure', 'troubleshoot', 'debug', 'error',
    'performance', 'optimize', 'tune', 'benchmark',
}


def classify_query(query: str) -> str:
    """
    Classify a user query to determine the best retrieval strategy.

    Args:
        query: The user's natural-language question.

    Returns:
        One of:
        - ``'structured'``: query is a factual GPU/version lookup
        - ``'hybrid'``: query mentions specific hardware but needs docs too
        - ``'semantic'``: general documentation question
    """
    import re

    query_lower = query.lower()

    has_gpu_ref = any(
        re.search(pat, query, re.IGNORECASE) for pat in _GPU_PATTERNS
    )
    has_version_ref = any(
        re.search(pat, query, re.IGNORECASE) for pat in _VERSION_PATTERNS
    )
    has_structured_kw = any(kw in query_lower for kw in _STRUCTURED_KEYWORDS)
    has_semantic_kw = any(kw in query_lower for kw in _SEMANTIC_KEYWORDS)

    # Pure structured: mentions GPU/version + asks about compatibility
    if (has_gpu_ref or has_version_ref) and has_structured_kw and not has_semantic_kw:
        return 'structured'

    # Hybrid: mentions specific hardware but also asks how-to
    if has_gpu_ref or has_version_ref:
        return 'hybrid'

    # Default: semantic search
    return 'semantic'


def classify_hardware_intent(query: str, env_context: str = "") -> Optional[str]:
    """Detect if the user is asking about a specific class of hardware."""
    query_lower = query.lower()
    env_lower = env_context.lower()
    
    # Explicit query overrides
    if any(kw in query_lower for kw in ['laptop', 'npu', 'ryzen', 'strix']):
        return "Ryzen AI"
    if any(kw in query_lower for kw in ['instinct', 'mi300', 'mi250', 'server', 'datacenter']):
        return "Instinct/Radeon"
        
    # Implicit environment fallbacks
    if 'ryzen' in env_lower or 'npu' in env_lower:
        return "Ryzen AI"
    if 'instinct' in env_lower or 'gfx9' in env_lower:
        return "Instinct/Radeon"
        
    return None


def smart_retrieve(
    query: str,
    collection,
    gpu_db: Optional[Dict] = None,
    embedding_model=None,
    top_k: int = 8,
    doc_type_filter: Optional[str] = None,
    hardware_target_filter: Optional[str] = None,
    use_reranker: bool = True,
) -> Dict:
    """
    Intelligent retrieval that routes queries to the best data source.

    For GPU/version questions, queries the structured GPU database first.
    For general documentation questions, uses semantic search over ChromaDB.
    For hybrid queries, combines both sources.

    Args:
        query: The user's natural-language question.
        collection: A ChromaDB collection handle.
        gpu_db: Optional loaded GPU database dict (from gpu_compat.load_gpu_database).
        embedding_model: Optional SentenceTransformer for query encoding.
        top_k: Number of documentation chunks to retrieve.
        doc_type_filter: Optional filter on document type.
        use_reranker: If True, apply Cross-Encoder reranking.

    Returns:
        A dict with keys:
        - ``structured_data``: str — structured facts from GPU database (may be empty)
        - ``documentation``: str — semantic search results formatted as context
        - ``source_urls``: List[str] — verified source URLs for citations
        - ``query_type``: str — the classification result
    """
    import re

    query_type = classify_query(query)
    structured_data = ""
    source_urls = []

    logger.info("Query classified as '%s': %s", query_type, query[:80])

    # --- Structured lookup ---
    if gpu_db and query_type in ('structured', 'hybrid'):
        try:
            from src.gpu_compat import (
                lookup_gpu,
                format_gpu_report,
                format_compatibility_matrix,
                get_install_command,
                check_compatibility,
            )

            structured_parts = []

            # Extract GPU references from the query
            gpu_matches = []
            for pat in _GPU_PATTERNS:
                matches = re.findall(pat, query, re.IGNORECASE)
                gpu_matches.extend(matches)

            # Try to look up each mentioned GPU
            seen_gpus = set()
            for match in gpu_matches:
                gpu_info = lookup_gpu(gpu_db, match)
                if gpu_info and gpu_info.get('gfx_id') not in seen_gpus:
                    gfx_id = gpu_info['gfx_id']
                    seen_gpus.add(gfx_id)
                    report = format_gpu_report(gpu_db, gfx_id)
                    structured_parts.append(report)

            # Extract ROCm version references
            rocm_matches = re.findall(
                r'(?:ROCm|rocm)\s*([\d.]+)', query, re.IGNORECASE
            )

            # Extract PyTorch version references
            pytorch_matches = re.findall(
                r'(?:PyTorch|torch)\s*([\d.]+)', query, re.IGNORECASE
            )

            # If asking about install commands
            if any(kw in query.lower() for kw in ['install', 'pip', 'command']):
                for pt_ver in pytorch_matches:
                    rocm_ver = rocm_matches[0] if rocm_matches else None
                    cmd = get_install_command(gpu_db, pt_ver, rocm_ver)
                    if cmd:
                        structured_parts.append(
                            f"**Install Command (PyTorch {pt_ver}):**\n```bash\n{cmd}\n```"
                        )

            # If asking about compatibility
            if any(kw in query.lower() for kw in ['compatible', 'support', 'work']):
                for gfx_id in seen_gpus:
                    rocm_ver = rocm_matches[0] if rocm_matches else None
                    pt_ver = pytorch_matches[0] if pytorch_matches else None
                    compat = check_compatibility(
                        gpu_db, gfx_id, rocm_ver, pt_ver
                    )
                    if compat.get('warnings'):
                        structured_parts.append(
                            "**Compatibility Warnings:**\n"
                            + "\n".join(f"- ⚠️ {w}" for w in compat['warnings'])
                        )
                    if compat.get('recommendations'):
                        structured_parts.append(
                            "**Recommendations:**\n"
                            + "\n".join(f"- 💡 {r}" for r in compat['recommendations'])
                        )

            # If no specific GPU found but version query, show matrix
            if not seen_gpus and (rocm_matches or pytorch_matches):
                matrix = format_compatibility_matrix(gpu_db)
                structured_parts.append(matrix)

            if structured_parts:
                structured_data = "\n\n".join(structured_parts)

            # Add source URLs from the database
            db_sources = gpu_db.get('source_urls', {})
            source_urls.extend(db_sources.values())

        except ImportError:
            logger.warning("gpu_compat module not available, skipping structured lookup")
        except Exception as e:
            logger.warning("Structured lookup failed: %s", e)

    # --- Semantic search (skip for pure structured if we got data) ---
    doc_context = ""
    results = []
    if query_type != 'structured' or not structured_data:
        results = retrieve(
            query=query,
            collection=collection,
            embedding_model=embedding_model,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
            hardware_target_filter=hardware_target_filter,
            use_reranker=use_reranker,
        )
        doc_context = format_context(results)

        # Collect source URLs from retrieved chunks
        for r in results:
            url = r.get('metadata', {}).get('source_url', '')
            if url and url not in source_urls:
                source_urls.append(url)

    return {
        'structured_data': structured_data,
        'documentation': doc_context,
        'source_urls': source_urls,
        'query_type': query_type,
        'results': results,
    }

