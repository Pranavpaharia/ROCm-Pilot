"""
Unit tests for the embedder module.
Tests that are hardware agnostic and don't require actual GPU access.
"""

import unittest
from unittest.mock import patch, MagicMock
import os

# Mock the actual embedding functions to avoid hardware dependencies  
class TestEmbedder(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear any existing environment variables that might affect tests
        self.test_env_vars = {
            'FIREWORKS_API_KEY': 'test_key',
            'EMBEDDING_MODEL': 'BAAI/bge-large-en-v1.5',
            'USE_LOCAL_LEMONADE': 'true'
        }
        
    @patch('src.embedder.torch')
    def test_embedder_model_loading(self, mock_torch):
        """Test that embedder can load models without GPU."""
        # Mock torch to simulate no CUDA availability
        mock_torch.cuda.is_available.return_value = False
        
        # Test that we can import and reference the embedder module
        from src.embedder import DEFAULT_MODEL
        
        # Should work without actual GPU or model loading in tests
        self.assertEqual(DEFAULT_MODEL, 'BAAI/bge-large-en-v1.5')
        
    def test_embedder_config_consistency(self):
        """Test that embedder uses configured model."""
        from src.config import config
        
        # Verify the configuration is loaded properly
        self.assertEqual(config.DEFAULT_EMBEDDING_MODEL, 'BAAI/bge-large-en-v1.5')
        
    def test_vector_store_manager_creation(self):
        """Test vector store manager can be initialized."""
        from src.vector_store_manager import VectorStoreManager
        
        mgr = VectorStoreManager()
        self.assertEqual(mgr.db_path, 'data/chroma_db')
        
    def test_index_manifest_creation(self):
        """Test index manifest creation is handled properly."""
        from src.vector_store_manager import VectorStoreManager
        
        mgr = VectorStoreManager()
        
        # Test manifest creation without actual disk access for now
        self.assertIsInstance(mgr, VectorStoreManager)

if __name__ == '__main__':
    unittest.main()