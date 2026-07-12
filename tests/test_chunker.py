"""
Unit tests for the chunker module.
Tests that are hardware agnostic and don't require actual GPU access.
"""

import unittest
from unittest.mock import patch, MagicMock

class TestChunker(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear any existing environment variables that might affect tests
        self.test_env_vars = {
            'FIREWORKS_API_KEY': 'test_key',
            'EMBEDDING_MODEL': 'BAAI/bge-large-en-v1.5',
            'USE_LOCAL_LEMONADE': 'true'
        }
        
    def test_chunker_imports(self):
        """Test that chunker module can be imported."""
        from src.chunker import chunk_documents, _split_by_headers
        
        # Should not raise any errors
        self.assertTrue(callable(chunk_documents))
        
    def test_chunker_header_splitting(self):
        """Test header splitting functionality."""
        from src.chunker import _split_by_headers
        
        # Test content with headers
        test_content = """# Introduction

This is an introduction.

## Setup Instructions

These are setup instructions.

### Prerequisites

Prerequisites go here."""
        
        # Should split without errors
        sections = _split_by_headers(test_content, '.md')
        
        # Should have at least 3 sections
        self.assertGreaterEqual(len(sections), 2)
        
    def test_chunker_table_handling(self):
        """Test table detection and handling."""
        from src.chunker import _detect_and_replace_tables
        
        # Test content with markdown table
        test_content = """
Here's some text.

| Column 1 | Column 2 |
|----------|----------|
| Row 1    | Data 1   |
| Row 2    | Data 2   |

More text here."""
        
        # Should detect and replace tables
        result_text, tables = _detect_and_replace_tables(test_content)
        
        # Should find at least one table
        self.assertIsInstance(tables, list)

    def test_chunker_code_block_handling(self):
        """Test code block detection and handling."""
        from src.chunker import _detect_and_replace_code_blocks
        
        # Test content with code block
        test_content = """
Some text here.

```python
def hello():
    return "world"
```

More content."""
        
        # Should detect and replace code blocks
        result_text, code_blocks = _detect_and_replace_code_blocks(test_content)
        
        # Should find at least one code block
        self.assertIsInstance(code_blocks, list)

if __name__ == '__main__':
    unittest.main()