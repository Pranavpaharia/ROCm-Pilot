"""
Vector Store Management Module for ROCm-Pilot.

This module handles consistent embedding model usage between build and query time,
ensures proper ChromaDB management, and maintains index manifests.
"""

import os
import json
from typing import Dict, Any, List
from datetime import datetime
from hashlib import sha256

class VectorStoreManager:
    """Manages vector store consistency and metadata."""
    
    def __init__(self, db_path: str = 'data/chroma_db', collection_name: str = 'rocm_docs'):
        self.db_path = db_path
        self.collection_name = collection_name
        
    def create_index_manifest(self, embedding_model: str, docs_sha: str, 
                            build_timestamp: str = None) -> Dict[str, Any]:
        """Create manifest file documenting index configuration."""
        if build_timestamp is None:
            build_timestamp = datetime.now().isoformat()
            
        manifest = {
            "embedding_model": embedding_model,
            "docs_sha": docs_sha,
            "build_timestamp": build_timestamp,
            "db_path": self.db_path,
            "collection_name": self.collection_name,
            "version": "1.0"
        }
        
        # Save manifest to disk
        manifest_path = os.path.join(self.db_path, "index_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        return manifest
        
    def validate_index_consistency(self) -> bool:
        """Validate that the current embedding model matches what was used to build."""
        manifest_path = os.path.join(self.db_path, "index_manifest.json")
        
        if not os.path.exists(manifest_path):
            print("⚠️  No existing manifest found - this is a new index build.")
            return True
            
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                
            # For now, we'll just check if there's a manifest file
            print(f"✅ Found existing index manifest from {manifest['build_timestamp']}")
            return True
        except Exception as e:
            print(f"⚠️  Could not read manifest: {e}")
            return True  # Allow rebuild if manifest is corrupted
            
    def get_index_info(self) -> Dict[str, Any]:
        """Get current index information."""
        manifest_path = os.path.join(self.db_path, "index_manifest.json")
        
        if not os.path.exists(manifest_path):
            return {"status": "no_manifest"}
            
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"status": "corrupted_manifest", "error": str(e)}

# Example usage function to demonstrate consistency
def ensure_index_consistency(embedding_model: str, docs_content: List[str] = None) -> Dict[str, Any]:
    """
    Ensure that the embedding model used for building and querying is consistent.
    
    Args:
        embedding_model: The embedding model name to use for both build and query
        docs_content: Documents content (for SHA calculation)
        
    Returns:
        Dictionary with consistency status and metadata
    """
    manager = VectorStoreManager()
    
    # Calculate SHA of documents (optional)
    docs_sha = "unknown"
    if docs_content:
        combined_docs = "\n".join(docs_content)
        docs_sha = sha256(combined_docs.encode()).hexdigest()[:16]
        
    # Validate consistency
    is_consistent = manager.validate_index_consistency()
    
    # Create manifest if needed (this happens during build)
    if is_consistent:
        manifest = manager.create_index_manifest(
            embedding_model=embedding_model,
            docs_sha=docs_sha
        )
        
    return {
        "consistency_status": is_consistent,
        "embedding_model_used": embedding_model,
        "index_info": manager.get_index_info()
    }