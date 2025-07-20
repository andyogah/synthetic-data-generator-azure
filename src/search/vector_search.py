from vector_db import VectorDB
from azure_search import AzureSearch

class VectorSearch:
    def __init__(self, vector_db: VectorDB, azure_search: AzureSearch):
        self.vector_db = vector_db
        self.azure_search = azure_search

    def vector_search(self, query_embedding, top_k=10):
        results = self.vector_db.query(query_embedding, top_k)
        return results

    def semantic_search(self, query, top_k=10):
        query_embedding = self.azure_search.embed_query(query)
        results = self.vector_search(query_embedding, top_k)
        return results

    def hybrid_search(self, query, top_k=10):
        text_results = self.azure_search.text_search(query, top_k)
        vector_results = self.semantic_search(query, top_k)
        
        combined_results = self._combine_results(text_results, vector_results)
        return combined_results

    def _combine_results(self, text_results, vector_results):
        combined = {result['id']: result for result in text_results}
        for result in vector_results:
            if result['id'] not in combined:
                combined[result['id']] = result
        return list(combined.values())

    def rerank_results(self, results, query):
        # Implement reranking logic based on relevance to the query
        return sorted(results, key=lambda x: x['score'], reverse=True)