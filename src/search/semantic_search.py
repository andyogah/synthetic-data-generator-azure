from typing import List
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.core.credentials import AzureKeyCredential
import numpy as np

class SemanticSearch:
    def __init__(self, search_service_name: str, index_name: str, api_key: str):
        self.search_client = SearchClient(
            endpoint=f"https://{search_service_name}.search.windows.net",
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        results = self.search_client.search(
            search_text=query,
            top=top_k,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="default"
        )
        return [result for result in results]

    def search_with_embeddings(self, query_embedding: np.ndarray, top_k: int = 10) -> List[dict]:
        # Assuming we have a method to convert embeddings to a searchable format
        # This is a placeholder for actual implementation
        results = self.search_client.search(
            search_text="*",
            top=top_k,
            filter=f"embedding eq {query_embedding.tolist()}"
        )
        return [result for result in results]

    def rerank_results(self, results: List[dict], query: str) -> List[dict]:
        # Placeholder for reranking logic based on semantic similarity
        # This could involve additional scoring based on the query
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int = 10) -> List[dict]:
        text_results = self.search(query, top_k)
        embedding_results = self.search_with_embeddings(query_embedding, top_k)
        
        combined_results = text_results + embedding_results
        reranked_results = self.rerank_results(combined_results, query)
        
        return reranked_results
