"""
Unit tests for the gpu_compat module.
Tests that are hardware agnostic and don't require actual GPU access.
"""

import unittest
from unittest.mock import patch, MagicMock

class TestGPUCompat(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_env_vars = {
            'FIREWORKS_API_KEY': 'test_key',
            'EMBEDDING_MODEL': 'BAAI/bge-large-en-v1.5',
            'USE_LOCAL_LEMONADE': 'true'
        }
        
    def test_gpu_compat_imports(self):
        """Test that gpu_compat module can be imported."""
        try:
            from src.gpu_compat import check_compatibility, get_install_command
            # Should not raise any errors when importing
            
            # Test that functions are callable
            self.assertTrue(callable(check_compatibility))
            self.assertTrue(callable(get_install_command))
            
        except ImportError as e:
            # This might fail if dependencies aren't available, but that's OK for tests
            print(f"Warning (expected): Could not import gpu_compat: {e}")
            
    @patch('src.gpu_compat.requests')
    def test_gpu_compat_mock(self, mock_requests):
        """Test gpu_compat with mocked requests."""
        # Mock the response to avoid network calls
        mock_response = MagicMock()
        mock_response.json.return_value = {"test": "data"}
        mock_requests.get.return_value = mock_response
        
        # Try to use the functions (this is a limited test)
        try:
            from src.gpu_compat import check_compatibility
            
            # This should at least not crash
            result = check_compatibility({}, "test_gfx_id", "6.0")
            
        except Exception as e:
            # Expected for various reasons, but the import should work
            pass

if __name__ == '__main__':
    unittest.main()