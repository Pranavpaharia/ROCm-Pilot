"""
Document collector for ROCm documentation.
Walks cloned GitHub repos and extracts all documentation files.
Includes markdown table extraction for compatibility matrices.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm


# File extensions to collect
DOC_EXTENSIONS = {'.md', '.rst', '.txt'}

# Directories to skip (not useful for RAG)
SKIP_DIRS = {
    '.git', '__pycache__', 'node_modules', '.github',
    '_build', 'build', 'dist', '.tox', 'venv',
    'site-packages', '_static', '_templates', '.eggs',
}

# Map repo directory names to their base URLs on rocm.docs.amd.com
REPO_URL_MAP = {
    'ROCm': 'https://rocm.docs.amd.com/en/latest/',
    'rocm-install-on-linux': 'https://rocm.docs.amd.com/projects/install-on-linux/en/latest/',
    'rocm-blogs': 'https://rocm.docs.amd.com/en/latest/blogs/',
    'gpuaidev': 'https://github.com/ROCm/gpuaidev/blob/main/',
}


def classify_doc_type(filepath: str) -> str:
    """Classify a document by its content type based on file path."""
    path_lower = filepath.lower()

    if any(kw in path_lower for kw in ['install', 'setup', 'getting-started', 'quick-start']):
        return 'installation'
    elif 'blog' in path_lower:
        return 'blog'
    elif any(kw in path_lower for kw in ['tutorial', 'how-to', 'howto', 'guide']):
        return 'tutorial'
    elif any(kw in path_lower for kw in ['api', 'reference']):
        return 'reference'
    elif any(kw in path_lower for kw in ['concept', 'architecture', 'overview']):
        return 'conceptual'
    elif 'example' in path_lower:
        return 'example'
    else:
        return 'general'


def _extract_markdown_tables(content: str) -> List[Dict]:
    """
    Extract GFM-style markdown tables from content.
    
    Returns a list of table dicts, each containing:
        - headers: List[str] (column headers)
        - rows: List[List[str]] (row data)
        - raw: str (original markdown text)
        - start_pos: int (character position in content)
        - end_pos: int (character position in content)
    """
    tables = []
    
    # Pattern to match markdown tables:
    # A header row with pipes, a separator row with dashes/pipes, then data rows
    table_pattern = re.compile(
        r'((?:^\|.+\|\s*\n)+)'  # header + data rows (captured together)
        r'(?=\n[^\n]*\n|\Z)',    # followed by non-table content or end
        re.MULTILINE,
    )
    
    # More precise: find table blocks
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        # Look for a line that starts with | (potential table start)
        if lines[i].strip().startswith('|'):
            table_lines = []
            start_pos = i
            
            # Collect consecutive pipe-delimited lines
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            
            # Need at least 3 lines: header, separator, 1+ data rows
            if len(table_lines) >= 3:
                # Verify second line is a separator (contains ---)
                separator = table_lines[1].strip()
                if re.match(r'^\|[\s\-:|]+\|$', separator):
                    # Parse the table
                    headers = _parse_table_row(table_lines[0])
                    rows = []
                    for row_line in table_lines[2:]:
                        row = _parse_table_row(row_line)
                        if row:
                            rows.append(row)
                    
                    raw_text = '\n'.join(table_lines)
                    tables.append({
                        'headers': headers,
                        'rows': rows,
                        'raw': raw_text,
                        'start_line': start_pos,
                        'end_line': i,
                    })
        else:
            i += 1
    
    return tables


def _parse_table_row(line: str) -> List[str]:
    """Parse a single markdown table row into cell values."""
    # Remove leading/trailing pipes and split
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    
    cells = [cell.strip() for cell in line.split('|')]
    return cells


def _format_table_as_text(table: Dict) -> str:
    """
    Convert a parsed table dict into a readable text format
    suitable for embedding and retrieval.
    
    Format: "Column1: Value1 | Column2: Value2 | ..."
    One line per row.
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


def collect_documents(raw_docs_dir: str) -> List[Dict]:
    """
    Walk through cloned repos and collect all documentation files.

    Args:
        raw_docs_dir: Path to the directory containing cloned repos.

    Returns:
        List of document dicts, each containing:
            - content: str (raw file text)
            - source_repo: str (e.g., "ROCm")
            - source_file: str (relative path within the repo)
            - source_url: str (estimated URL on rocm.docs.amd.com)
            - doc_type: str (installation, blog, tutorial, reference, etc.)
    """
    documents = []
    raw_docs_path = Path(raw_docs_dir)

    if not raw_docs_path.exists():
        raise FileNotFoundError(
            f"Raw docs directory not found: {raw_docs_dir}\n"
            "Run setup.sh first to clone the documentation repos."
        )

    for repo_dir in sorted(raw_docs_path.iterdir()):
        if not repo_dir.is_dir():
            continue

        repo_name = repo_dir.name
        base_url = REPO_URL_MAP.get(repo_name, f'https://github.com/ROCm/{repo_name}/')

        # Collect all matching files
        doc_files = []
        for root, dirs, files in os.walk(repo_dir):
            # Prune irrelevant directories in-place
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for filename in files:
                if Path(filename).suffix.lower() in DOC_EXTENSIONS:
                    doc_files.append(os.path.join(root, filename))

        # Read each file
        for filepath in tqdm(doc_files, desc=f"📂 {repo_name}", unit="files"):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Skip trivially small files (likely just headers or redirects)
                if len(content.strip()) < 100:
                    continue

                # Truncate very large files (auto-generated API docs, etc.)
                if len(content) > 50_000:
                    content = content[:50_000]

                rel_path = os.path.relpath(filepath, repo_dir)
                source_url = base_url + rel_path.replace('.rst', '.html').replace('.md', '.html')

                # Extract markdown tables if present
                tables = _extract_markdown_tables(content)
                
                documents.append({
                    'content': content,
                    'source_repo': repo_name,
                    'source_file': rel_path,
                    'source_url': source_url,
                    'doc_type': classify_doc_type(rel_path),
                    'tables': tables,
                })

            except Exception as e:
                print(f"  ⚠️  Could not read {filepath}: {e}")

    print(f"\n✅ Collected {len(documents)} documents from {raw_docs_dir}")
    return documents


if __name__ == '__main__':
    docs = collect_documents('data/raw_docs')
    # Print summary by type
    doc_types = {}
    for d in docs:
        doc_types[d['doc_type']] = doc_types.get(d['doc_type'], 0) + 1
    print("\nDocument breakdown:")
    for dtype, count in sorted(doc_types.items()):
        print(f"  {dtype}: {count}")
