from typing import List, Any
from .text_search import TextSearch
from .vector_search import VectorSearch
from .semantic_search import SemanticSearch

class HybridSearch:
    def __init__(self, text_search: TextSearch, vector_search: VectorSearch, semantic_search: SemanticSearch):
        self.text_search = text_search
        self.vector_search = vector_search
        self.semantic_search = semantic_search

    def search(self, query: str, use_text: bool = True, use_vector: bool = True, use_semantic: bool = True) -> List[Any]:
        results = []
        
        if use_text:
            text_results = self.text_search.search(query)
            results.extend(text_results)

        if use_vector:
            vector_results = self.vector_search.search(query)
            results.extend(vector_results)

        if use_semantic:
            semantic_results = self.semantic_search.search(query)
            results.extend(semantic_results)

        # Reranking logic can be added here if needed
        # For now, we will just return the combined results
        return results

    def rerank_results(self, results: List[Any]) -> List[Any]:
        # Implement reranking logic based on specific criteria
        # This is a placeholder for future implementation
        return results  # Currently returns results as is
