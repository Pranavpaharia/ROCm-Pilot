"""
Version-aware Retrieval System for ROCm-Pilot.
Prefers chunks matching detected ROCm major version; hybrid BM25+dense if time.
"""

import re
from typing import List, Dict, Any
from collections import defaultdict

class VersionAwareRetriever:
    """Retrieval system that prioritizes chunks matching ROCm version."""
    
    def __init__(self):
        self.version_pattern = re.compile(r'roc?m\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)
        
    def detect_rocm_version_from_query(self, query: str) -> str:
        """Detect ROCm version from the user's question."""
        # Look for ROCm major versions in query
        matches = self.version_pattern.findall(query)
        
        if not matches:
            # Try to extract from context or use default
            return "6.0"  # Default fallback
            
        # Return the first detected version, preferring major versions
        return matches[0]
        
    def prioritize_by_rocm_version(self, results: List[Dict], query: str) -> List[Dict]:
        """Prioritize retrieval results based on ROCm version match."""
        
        detected_version = self.detect_rocm_version_from_query(query)
        if not detected_version:
            return results
            
        # Create version-based prioritization
        priority_results = []
        
        # Separate results by version match
        matching_version_results = []
        non_matching_results = []
        
        for result in results:
            # Check if the result mentions the detected ROCm version
            content = (result.get('text', '') + 
                      result.get('metadata', {}).get('source_url', ''))
            
            # Case insensitive match for the detected version
            if f"roc{detected_version}" in content.lower() or \
               f"rocm {detected_version}" in content.lower():
                matching_version_results.append(result)
            else:
                non_matching_results.append(result)
                
        # Prioritize matching version results first
        priority_results.extend(matching_version_results)
        priority_results.extend(non_matching_results) 
        
        return priority_results
        
    def hybrid_retrieval(self, query: str, semantic_results: List[Dict], 
                        bm25_results: List[Dict] = None) -> List[Dict]:
        """Hybrid BM25 + dense retrieval with version prioritization."""
        
        # First, apply ROCm version prioritization to semantic results 
        if semantic_results:
            semantic_results = self.prioritize_by_rocm_version(semantic_results, query)
            
        # If we have BM25 results and want to merge them
        if bm25_results:
            # Simple merge with version prioritization for semantic results
            combined = []
            semantic_idx = 0
            
            # Interleave version-prioritized semantic results with BM25 if available
            for i, result in enumerate(semantic_results):
                combined.append(result)
                
                # Add BM25 results if available
                if i < len(bm25_results) and bm25_results[i]:
                    combined.append(bm25_results[i])
                    
            return combined
            
        # If no BM25, just return version-prioritized semantic results
        return semantic_results

    def analyze_query_and_prioritize(self, query: str, all_results: List[Dict]) -> Dict[str, Any]:
        """Analyze query and return prioritized results with metadata."""
        
        detected_version = self.detect_rocm_version_from_query(query)
        
        # Apply prioritization
        prioritized_results = self.prioritize_by_rocm_version(all_results, query)
        
        return {
            "query": query,
            "detected_rocm_version": detected_version,
            "prioritized_results": prioritized_results,
            "total_results": len(all_results),
            "version_matched_results": sum(1 for r in all_results 
                                        if detected_version and 
                                        (detected_version in str(r.get('text', '')).lower() or
                                         detected_version in str(r.get('metadata', {}).get('source_url', '')).lower()))
        }

# Export main function for easy use
def prioritize_rocm_version_results(query: str, results: List[Dict]) -> Dict[str, Any]:
    """Main function to prioritize ROCm version-specific results."""
    retriever = VersionAwareRetriever()
    return retriever.analyze_query_and_prioritize(query, results)

if __name__ == "__main__":
    print("=== Version-aware Retrieval System ===")
    
    # Example usage
    retriever = VersionAwareRetriever()
    
    test_results = [
        {
            "text": "Installation guide for ROCm 6.0",
            "metadata": {"source_url": "/docs/install_6.0.md"}
        },
        {
            "text": "ROCm 5.7 compatibility matrix",
            "metadata": {"source_url": "/docs/compat_5.7.md"} 
        },
        {
            "text": "ROCm 6.0 API documentation",
            "metadata": {"source_url": "/docs/api_6.0.md"}
        }
    ]
    
    query = "How to install ROCm 6.0 on Ubuntu"
    result = prioritize_rocm_version_results(query, test_results)
    
    print(f"Query: {result['query']}")
    print(f"Detected ROCm version: {result['detected_rocm_version']}")
    print(f"Total results: {result['total_results']}")
    print("Prioritized results:")
    for i, r in enumerate(result['prioritized_results']):
        print(f"  {i+1}. {r['text']}")