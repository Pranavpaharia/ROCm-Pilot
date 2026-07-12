"""
Test configuration and mock environment setup for ROCm-Pilot.
This file provides hardware-agnostic test configurations and mocks.
"""

import os
from unittest.mock import patch, MagicMock

# Mock configuration for test environments  
MOCK_ENV_VARS = {
    'FIREWORKS_API_KEY': 'test_api_key',
    'EMBEDDING_MODEL': 'BAAI/bge-large-en-v1.5',
    'USE_LOCAL_LEMONADE': 'true',
    'CHROMA_DB_PATH': '/tmp/test_chroma_db',
    'USE_GPU': 'false'
}

def setup_mock_environment():
    """Set up mock environment for testing without hardware dependencies."""
    # Clear any existing test-specific environment variables
    for key in MOCK_ENV_VARS:
        if key in os.environ:
            del os.environ[key]
    
    # Set mock environment variables
    for key, value in MOCK_ENV_VARS.items():
        os.environ[key] = value
        
def create_test_config():
    """Create a test configuration object."""
    from src.config import Config
    
    # Override some settings for testing
    class TestConfig(Config):
        CHROMA_DB_PATH = '/tmp/test_chroma_db'
        COLLECTION_NAME = 'test_rocm_docs'
        USE_LOCAL_LEMONADE = True
        USE_GPU = False
        
    return TestConfig()

# Mock system components for testing  
def mock_system_components():
    """Mock hardware-specific components for testing."""
    mocks = {}
    
    # Mock GPU detection
    with patch('torch.cuda.is_available') as mock_cuda:
        mock_cuda.return_value = False
        
    # Mock GPU-specific functionality
    with patch('src.gpu_compat.check_compatibility') as mock_compat:
        mock_compat.return_value = {
            'warnings': [],
            'recommendations': ['Test recommendation']
        }
    
    return mocks

if __name__ == "__main__":
    print("Test configuration module loaded successfully")