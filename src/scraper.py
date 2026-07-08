"""
Document collector for ROCm documentation.
Walks cloned GitHub repos and extracts all documentation files.
"""

import os
from pathlib import Path
from typing import List, Dict
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

                documents.append({
                    'content': content,
                    'source_repo': repo_name,
                    'source_file': rel_path,
                    'source_url': source_url,
                    'doc_type': classify_doc_type(rel_path),
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
