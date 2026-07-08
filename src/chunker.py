"""
Document chunker for ROCm documentation.
Splits documents into overlapping chunks suitable for embedding and retrieval.
Preserves markdown tables as structured blocks that don't get split across chunks.
"""

import re
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


# Placeholder token for tables during chunking
TABLE_PLACEHOLDER = "<<TABLE_BLOCK_{idx}>>"


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


def _detect_and_replace_tables(text: str) -> Tuple[str, List[Dict]]:
    """
    Detect markdown tables in text and replace them with placeholder tokens.
    
    Returns:
        - modified text with placeholders
        - list of table dicts with their raw content and placeholder index
    """
    tables_found = []
    lines = text.split('\n')
    result_lines = []
    i = 0
    table_idx = 0
    
    while i < len(lines):
        # Check if this line starts a table
        if lines[i].strip().startswith('|'):
            table_lines = []
            start_i = i
            
            # Collect consecutive pipe-delimited lines
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            
            # Validate: need header + separator + at least 1 data row
            if len(table_lines) >= 3:
                separator = table_lines[1].strip()
                if re.match(r'^\|[\s\-:|]+\|$', separator):
                    # Valid table — replace with placeholder
                    raw = '\n'.join(table_lines)
                    headers = _parse_table_row(table_lines[0])
                    rows = []
                    for row_line in table_lines[2:]:
                        row = _parse_table_row(row_line)
                        if row:
                            rows.append(row)
                    
                    placeholder = TABLE_PLACEHOLDER.format(idx=table_idx)
                    result_lines.append(placeholder)
                    tables_found.append({
                        'placeholder': placeholder,
                        'headers': headers,
                        'rows': rows,
                        'raw': raw,
                        'index': table_idx,
                    })
                    table_idx += 1
                    continue
            # Not a valid table, keep lines as-is
            result_lines.extend(table_lines)
        else:
            result_lines.append(lines[i])
            i += 1
    
    return '\n'.join(result_lines), tables_found


def _parse_table_row(line: str) -> List[str]:
    """Parse a single markdown table row into cell values."""
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    cells = [cell.strip() for cell in line.split('|')]
    return cells


def _format_table_for_embedding(table: Dict) -> str:
    """
    Convert a parsed table into a readable text format for embedding.
    
    Format: "Header1: Value1 | Header2: Value2 | ..."
    One line per data row.
    """
    headers = table.get('headers', [])
    rows = table.get('rows', [])
    
    if not headers or not rows:
        return table.get('raw', '')
    
    lines = []
    for row in rows:
        parts = []
        for j, cell in enumerate(row):
            if j < len(headers):
                parts.append(f"{headers[j]}: {cell}")
            else:
                parts.append(cell)
        lines.append(' | '.join(parts))
    
    return '\n'.join(lines)


def _restore_tables_in_chunk(chunk_text: str, tables: List[Dict]) -> str:
    """
    Replace table placeholders in a chunk with formatted table text.
    """
    for table in tables:
        placeholder = table['placeholder']
        if placeholder in chunk_text:
            formatted = _format_table_for_embedding(table)
            chunk_text = chunk_text.replace(placeholder, f"\n{formatted}\n")
    return chunk_text


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


def _chunk_text_preserving_tables(
    text: str,
    tables: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into chunks while keeping table blocks intact.
    
    Strategy:
    1. Split text around table placeholders into segments
    2. Chunk each non-table segment normally
    3. Attach tables as complete blocks to the nearest chunk
    """
    if not tables:
        return _chunk_text(text, chunk_size, overlap)
    
    # Split text into segments around placeholders
    segments = []
    remaining = text
    
    for table in sorted(tables, key=lambda t: t['index']):
        placeholder = table['placeholder']
        if placeholder in remaining:
            before, after = remaining.split(placeholder, 1)
            if before.strip():
                segments.append({'type': 'text', 'content': before})
            segments.append({'type': 'table', 'table': table})
            remaining = after
    
    if remaining.strip():
        segments.append({'type': 'text', 'content': remaining})
    
    # Build chunks from segments
    chunks = []
    current_chunk_words = []
    current_chunk_tables = []
    
    for segment in segments:
        if segment['type'] == 'table':
            # Tables get attached to the current chunk or start a new one
            formatted = _format_table_for_embedding(segment['table'])
            table_words = len(formatted.split())
            
            if table_words > chunk_size:
                # Very large table — flush current chunk, then add table as its own chunk
                if current_chunk_words:
                    chunks.append(' '.join(current_chunk_words))
                    current_chunk_words = []
                chunks.append(formatted)
            elif len(current_chunk_words) + table_words > chunk_size:
                # Flush current chunk, start new one with table
                if current_chunk_words:
                    chunks.append(' '.join(current_chunk_words))
                current_chunk_words = formatted.split()
            else:
                # Add table to current chunk
                current_chunk_words.extend(formatted.split())
        else:
            # Text segment — chunk normally
            text_content = segment['content']
            text_chunks = _chunk_text(text_content, chunk_size, overlap)
            
            for i, tc in enumerate(text_chunks):
                tc_words = tc.split()
                if len(current_chunk_words) + len(tc_words) <= chunk_size:
                    current_chunk_words.extend(tc_words)
                else:
                    if current_chunk_words:
                        chunks.append(' '.join(current_chunk_words))
                    # If this is the first sub-chunk, start fresh
                    current_chunk_words = tc_words
    
    # Don't forget the last chunk
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))
    
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
    Markdown tables are preserved as intact blocks within chunks.

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
            - has_table: bool (whether chunk contains table data)
    """
    all_chunks = []

    for doc in tqdm(documents, desc="✂️  Chunking", unit="docs"):
        content = _clean_content(doc['content'])

        file_ext = '.rst' if doc['source_file'].endswith('.rst') else '.md'
        sections = _split_by_headers(content, file_ext)

        for section in sections:
            # Detect and replace tables with placeholders
            modified_text, tables = _detect_and_replace_tables(section['content'])
            
            # Chunk with table preservation
            text_chunks = _chunk_text_preserving_tables(
                modified_text, tables, chunk_size, overlap,
            )

            for i, chunk_text in enumerate(text_chunks):
                # Restore any table placeholders with formatted text
                chunk_text = _restore_tables_in_chunk(chunk_text, tables)
                
                # Skip very short chunks (< 20 words)
                if len(chunk_text.split()) < 20:
                    continue

                # Check if chunk contains table data
                has_table = any(
                    _format_table_for_embedding(t) in chunk_text
                    for t in tables
                ) if tables else False

                all_chunks.append({
                    'text': chunk_text,
                    'source_repo': doc['source_repo'],
                    'source_file': doc['source_file'],
                    'source_url': doc['source_url'],
                    'doc_type': doc['doc_type'],
                    'section_title': section['title'],
                    'chunk_index': i,
                    'has_table': has_table,
                })

    print(f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


if __name__ == '__main__':
    from src.scraper import collect_documents

    docs = collect_documents('data/raw_docs')
    chunks = chunk_documents(docs)

    # Print summary
    table_chunks = sum(1 for c in chunks if c.get('has_table'))
    print(f"\nChunks with tables: {table_chunks}/{len(chunks)}")

    # Print a sample chunk
    if chunks:
        sample = chunks[0]
        print("\n--- Sample Chunk ---")
        print(f"Source: {sample['source_repo']}/{sample['source_file']}")
        print(f"Section: {sample['section_title']}")
        print(f"Type: {sample['doc_type']}")
        print(f"Has table: {sample.get('has_table', False)}")
        print(f"Words: {len(sample['text'].split())}")
        print(f"Text: {sample['text'][:300]}...")