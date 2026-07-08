"""
Document chunker for ROCm documentation.
Splits documents into overlapping chunks suitable for embedding and retrieval.
"""

import re
from typing import List, Dict
from tqdm import tqdm


def _split_by_headers(content: str, file_ext: str) -> List[Dict[str, str]]:
    """Split document content by markdown or RST headers into sections."""
    sections = []

    if file_ext == '.rst':
        # RST headers: a title line followed by a line of =, -, ~, ^, " characters
        pattern = r'(?=\n[^\n]+\n[=\-~\^\"]+\n)'
    else:
        # Markdown headers: lines starting with one or more #
        pattern = r'(?=\n#{1,4}\s)'

    parts = re.split(pattern, content)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract section title from the first line
        lines = part.split('\n')
        title = lines[0].strip().lstrip('#').strip()

        # For RST, clean up the underline character row
        if len(lines) > 1 and re.match(r'^[=\-~\^\"]+$', lines[1].strip()):
            title = lines[0].strip()

        sections.append({
            'title': title[:200],
            'content': part,
        })

    # If no headers were found, treat the whole content as one section
    if not sections:
        sections = [{'title': 'Document', 'content': content}]

    return sections


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by approximate word count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
        if start >= len(words):
            break

    return chunks


def _clean_content(text: str) -> str:
    """Strip RST/Markdown formatting artifacts that add noise."""
    # Remove RST directives (e.g., .. note::, .. code-block::)
    text = re.sub(r'\.\.\s+\w+::[^\n]*\n', '\n', text)
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove RST comment blocks
    text = re.sub(r'\.\.\s*\n\s+[^\n]+', '', text)
    return text.strip()


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """
    Process documents into overlapping chunks suitable for embedding.

    Args:
        documents: List of document dicts from scraper.collect_documents().
        chunk_size: Target number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of chunk dicts, each containing:
            - text: str (the chunk content)
            - source_repo: str
            - source_file: str
            - source_url: str
            - doc_type: str
            - section_title: str
            - chunk_index: int (index within the section)
    """
    all_chunks = []

    for doc in tqdm(documents, desc="✂️  Chunking", unit="docs"):
        content = _clean_content(doc['content'])

        file_ext = '.rst' if doc['source_file'].endswith('.rst') else '.md'
        sections = _split_by_headers(content, file_ext)

        for section in sections:
            text_chunks = _chunk_text(section['content'], chunk_size, overlap)

            for i, chunk_text in enumerate(text_chunks):
                # Skip very short chunks (< 20 words)
                if len(chunk_text.split()) < 20:
                    continue

                all_chunks.append({
                    'text': chunk_text,
                    'source_repo': doc['source_repo'],
                    'source_file': doc['source_file'],
                    'source_url': doc['source_url'],
                    'doc_type': doc['doc_type'],
                    'section_title': section['title'],
                    'chunk_index': i,
                })

    print(f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


if __name__ == '__main__':
    from src.scraper import collect_documents

    docs = collect_documents('data/raw_docs')
    chunks = chunk_documents(docs)

    # Print a sample chunk
    if chunks:
        sample = chunks[0]
        print("\n--- Sample Chunk ---")
        print(f"Source: {sample['source_repo']}/{sample['source_file']}")
        print(f"Section: {sample['section_title']}")
        print(f"Type: {sample['doc_type']}")
        print(f"Words: {len(sample['text'].split())}")
        print(f"Text: {sample['text'][:300]}...")
