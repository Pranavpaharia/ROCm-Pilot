"""
Configuration management for ROCm-Pilot.
Handles all configuration settings including model IDs, API endpoints,
embedding models, and other system parameters.
"""

import os
from typing import Optional

class Config:
    """Central configuration class for ROCm-Pilot system."""
    
    # --- Model Configuration ---
    DEFAULT_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')
    DEFAULT_LLM_MODEL = os.getenv('LLM_MODEL', 'omlx/qwen3-coder-30b')
    
    # --- API Configuration ---
    FIREWORKS_API_KEY = os.getenv('FIREWORKS_API_KEY', 'fw_Frwm71myTP8unXeR7Jxmcx')
    LEMONADE_URL = os.getenv('LEMONADE_URL', 'http://localhost:4000/v1')
    
    # --- Vector Store Configuration ---
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', 'data/chroma_db')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'rocm_docs')
    
    # --- Retrieval Configuration ---
    TOP_K = int(os.getenv('TOP_K', 8))
    
    # --- Local/Remote Configuration ---
    USE_LOCAL_LEMONADE = os.getenv('USE_LOCAL_LEMONADE', 'true').lower() == 'true'
    
    # --- GPU Configuration ---
    USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'
    
    @classmethod
    def get_config_dict(cls):
        """Return all configuration as a dictionary for logging/display."""
        return {
            'embedding_model': cls.DEFAULT_EMBEDDING_MODEL,
            'llm_model': cls.DEFAULT_LLM_MODEL,
            'chroma_db_path': cls.CHROMA_DB_PATH,
            'collection_name': cls.COLLECTION_NAME,
            'top_k': cls.TOP_K,
            'use_local_lemonade': cls.USE_LOCAL_LEMONADE,
            'use_gpu': cls.USE_GPU
        }

# Create a global config instance for easy access  
config = Config()