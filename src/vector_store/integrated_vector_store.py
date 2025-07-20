from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery, VectorQuery
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, SimpleField,
    SearchableField, VectorSearch, VectorSearchProfile,
    HnswAlgorithmConfiguration, VectorSearchAlgorithmKind,
    SemanticConfiguration, SemanticPrioritizedFields, SemanticField,
    SemanticSearch, VectorSearchAlgorithmMetric
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime
from .base_vector_store import VectorStoreInterface, SearchType
from ..config.settings import settings

logger = logging.getLogger(__name__)

class IntegratedVectorStore(VectorStoreInterface):
    """Azure AI Search with integrated vectorization"""
    
    def __init__(self):
        self.endpoint = settings.azure_search_endpoint
        self.api_key = settings.azure_search_api_key
        self.index_name = settings.azure_search_index_name
        
        credential = AzureKeyCredential(self.api_key)
        
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=credential,
            api_version=settings.azure_search_api_version
        )
        
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=credential,
            api_version=settings.azure_search_api_version
        )
        
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Create or update search index"""
        try:
            existing_index = self.index_client.get_index(self.index_name)
            logger.info(f"Index '{self.index_name}' already exists")
        except ResourceNotFoundError:
            logger.info(f"Creating new index '{self.index_name}'")
            self._create_index()
        except Exception as e:
            logger.error(f"Error checking index existence: {e}")
            raise
    
    def _create_index(self):
        """Create search index with integrated vectorization"""
        # Define index fields
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
            SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=settings.embedding_dimension,
                vector_search_profile_name="default-vector-profile"
            ),
            SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="metadata", type=SearchFieldDataType.String)
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="default-hnsw-algorithm",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": VectorSearchAlgorithmMetric.COSINE
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="default-vector-profile",
                    algorithm_configuration_name="default-hnsw-algorithm"
                )
            ]
        )
        
        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="default-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Create index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        try:
            self.index_client.create_index(index)
            logger.info(f"Successfully created index '{self.index_name}'")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index documents with integrated vectorization"""
        try:
            processed_docs = []
            
            for doc in documents:
                content = doc.get('content', '')
                
                if len(content) > settings.chunk_size:
                    # Split into chunks
                    chunks = self._chunk_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        processed_doc = {
                            'id': f"{doc['id']}_chunk_{i}",
                            'content': chunk,
                            'title': doc.get('title', f"Document {doc['id']} - Chunk {i}"),
                            'document_id': doc['id'],
                            'chunk_index': i,
                            'source': doc.get('source', 'unknown'),
                            'category': doc.get('category', 'general'),
                            'created_at': datetime.utcnow().isoformat(),
                            'metadata': json.dumps(doc.get('metadata', {}))
                        }
                        processed_docs.append(processed_doc)
                else:
                    processed_doc = {
                        'id': doc['id'],
                        'content': content,
                        'title': doc.get('title', f"Document {doc['id']}"),
                        'document_id': doc['id'],
                        'chunk_index': 0,
                        'source': doc.get('source', 'unknown'),
                        'category': doc.get('category', 'general'),
                        'created_at': datetime.utcnow().isoformat(),
                        'metadata': json.dumps(doc.get('metadata', {}))
                    }
                    processed_docs.append(processed_doc)
            
            # Upload documents (Azure AI Search handles vectorization automatically)
            result = self.search_client.upload_documents(processed_docs)
            
            success_count = sum(1 for r in result if r.succeeded)
            failed_count = len(result) - success_count
            
            logger.info(f"Indexed {success_count} documents successfully, {failed_count} failed")
            
            return {
                'success_count': success_count,
                'failed_count': failed_count,
                'total_processed': len(processed_docs),
                'approach': 'integrated',
                'results': result
            }
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    def search(self, query: str, search_type: SearchType = SearchType.HYBRID, 
               top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents with specified search type"""
        try:
            filter_expression = None
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_parts.append(f"{key} eq '{value}'")
                    else:
                        filter_parts.append(f"{key} eq {value}")
                filter_expression = " and ".join(filter_parts)
            
            if search_type == SearchType.TEXT:
                return self._text_search(query, top_k, filter_expression)
            elif search_type == SearchType.VECTOR:
                return self._vector_search(query, top_k, filter_expression)
            elif search_type == SearchType.SEMANTIC:
                return self._semantic_search(query, top_k, filter_expression)
            elif search_type == SearchType.HYBRID:
                return self._hybrid_search(query, top_k, filter_expression)
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def _text_search(self, query: str, top_k: int, filter_expr: Optional[str]) -> List[Dict[str, Any]]:
        """Full-text search"""
        results = self.search_client.search(
            search_text=query,
            top=top_k,
            filter=filter_expr,
            include_total_count=True
        )
        return [dict(result) for result in results]
    
    def _vector_search(self, query: str, top_k: int, filter_expr: Optional[str]) -> List[Dict[str, Any]]:
        """Vector search using query vectorization"""
        from ..data_processing.embedder import Embedder
        
        embedder = Embedder()
        query_vector = embedder.embed_text(query)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k,
            filter=filter_expr
        )
        return [dict(result) for result in results]
    
    def _semantic_search(self, query: str, top_k: int, filter_expr: Optional[str]) -> List[Dict[str, Any]]:
        """Semantic search with L2 ranking"""
        results = self.search_client.search(
            search_text=query,
            top=top_k,
            filter=filter_expr,
            query_type="semantic",
            semantic_configuration_name="default-semantic-config",
            query_caption="extractive",
            query_answer="extractive"
        )
        return [dict(result) for result in results]
    
    def _hybrid_search(self, query: str, top_k: int, filter_expr: Optional[str]) -> List[Dict[str, Any]]:
        """Hybrid search combining text and vector search"""
        from ..data_processing.embedder import Embedder
        
        embedder = Embedder()
        query_vector = embedder.embed_text(query)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        search_params = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "filter": filter_expr
        }
        
        if settings.enable_semantic_search:
            search_params.update({
                "query_type": "semantic",
                "semantic_configuration_name": "default-semantic-config",
                "query_caption": "extractive",
                "query_answer": "extractive"
            })
        
        results = self.search_client.search(**search_params)
        return [dict(result) for result in results]
    
    def _chunk_text(self, text: str) -> List[str]:
        """Simple text chunking with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + settings.chunk_size
            
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - settings.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document and all its chunks"""
        try:
            # Search for all chunks of the document
            results = self.search_client.search(
                search_text="*",
                filter=f"document_id eq '{doc_id}'",
                select="id"
            )
            
            # Delete all chunks
            docs_to_delete = [{"id": result["id"]} for result in results]
            
            if docs_to_delete:
                delete_results = self.search_client.delete_documents(docs_to_delete)
                success_count = sum(1 for r in delete_results if r.succeeded)
                logger.info(f"Deleted {success_count} chunks for document {doc_id}")
                return success_count > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents indexed"""
        try:
            results = self.search_client.search(
                search_text="*",
                include_total_count=True,
                top=0
            )
            return results.get_count()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store"""
        try:
            # Test index access
            index_stats = self.index_client.get_index_statistics(self.index_name)
            
            return {
                'status': 'healthy',
                'approach': 'integrated',
                'index_name': self.index_name,
                'document_count': index_stats.document_count,
                'storage_size': index_stats.storage_size
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'approach': 'integrated',
                'error': str(e)
            }