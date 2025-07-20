from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from .base_vector_store import VectorStoreInterface, SearchType
from .vector_db import VectorDB
from ..data_processing.embedder import Embedder
from ..data_processing.chunker import Chunker
from ..config.settings import settings

logger = logging.getLogger(__name__)

class CustomVectorStore(VectorStoreInterface):
    """Custom vector store using separate Azure services"""
    
    def __init__(self):
        self.vector_db = VectorDB(
            blob_connection_string=settings.azure_blob_connection_string,
            cosmos_connection_string=settings.azure_cosmos_connection_string,
            database_name=settings.cosmos_database_name,
            container_name=settings.cosmos_container_name
        )
        self.embedder = Embedder()
        self.chunker = Chunker()
        logger.info("Initialized custom vector store with separate Azure services")
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index documents with custom chunking and embedding"""
        try:
            success_count = 0
            failed_count = 0
            total_chunks = 0
            errors = []
            
            for doc in documents:
                try:
                    content = doc.get('content', '')
                    
                    # Chunk the content
                    if len(content) > settings.chunk_size:
                        chunks = self.chunker.chunk_text(content, settings.chunk_size, settings.chunk_overlap)
                    else:
                        chunks = [content]
                    
                    # Process each chunk
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{doc['id']}_chunk_{i}"
                        
                        # Generate embedding
                        embedding = self.embedder.embed_text(chunk)
                        
                        # Prepare metadata
                        metadata = {
                            'content': chunk,
                            'title': doc.get('title', f"Document {doc['id']} - Chunk {i}"),
                            'document_id': doc['id'],
                            'chunk_index': i,
                            'source': doc.get('source', 'unknown'),
                            'category': doc.get('category', 'general'),
                            'created_at': datetime.utcnow().isoformat(),
                            'metadata': doc.get('metadata', {})
                        }
                        
                        # Store in vector database
                        self.vector_db.store_vector_with_metadata(chunk_id, embedding, metadata)
                        total_chunks += 1
                    
                    success_count += 1
                    logger.info(f"Successfully indexed document: {doc['id']} with {len(chunks)} chunks")
                    
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Failed to index document {doc['id']}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            return {
                'success_count': success_count,
                'failed_count': failed_count,
                'total_chunks': total_chunks,
                'approach': 'custom',
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    def search(self, query: str, search_type: SearchType = SearchType.HYBRID, 
               top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents with specified search type"""
        try:
            if search_type == SearchType.TEXT:
                return self._text_search(query, top_k, filters)
            elif search_type == SearchType.VECTOR:
                return self._vector_search(query, top_k, filters)
            elif search_type == SearchType.SEMANTIC:
                return self._semantic_search(query, top_k, filters)
            elif search_type == SearchType.HYBRID:
                return self._hybrid_search(query, top_k, filters)
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def _text_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Text-based search using content matching"""
        results = self.vector_db.text_search(query, top_k, filters)
        return self._format_results(results)
    
    def _vector_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Vector similarity search"""
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_db.search_vectors(query_embedding, top_k, filters)
        return self._format_results(results)
    
    def _semantic_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Semantic search using enhanced vector similarity"""
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_db.semantic_search(query_embedding, query, top_k, filters)
        return self._format_results(results)
    
    def _hybrid_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Hybrid search combining text and vector approaches"""
        # Get vector results
        query_embedding = self.embedder.embed_text(query)
        vector_results = self.vector_db.search_vectors(query_embedding, top_k, filters)
        
        # Get text results
        text_results = self.vector_db.text_search(query, top_k, filters)
        
        # Combine and rerank results
        combined_results = self._combine_and_rerank(vector_results, text_results, top_k)
        
        return self._format_results(combined_results)
    
    def _combine_and_rerank(self, vector_results: List[Dict], text_results: List[Dict], top_k: int) -> List[Dict]:
        """Combine vector and text results with reranking"""
        # Simple combination strategy - can be enhanced
        all_results = {}
        
        # Add vector results with vector score
        for result in vector_results:
            doc_id = result['id']
            all_results[doc_id] = result
            all_results[doc_id]['vector_score'] = result.get('similarity', 0)
            all_results[doc_id]['text_score'] = 0
        
        # Add text results with text score
        for result in text_results:
            doc_id = result['id']
            if doc_id in all_results:
                all_results[doc_id]['text_score'] = result.get('score', 0)
            else:
                all_results[doc_id] = result
                all_results[doc_id]['vector_score'] = 0
                all_results[doc_id]['text_score'] = result.get('score', 0)
        
        # Calculate combined score
        for doc_id, result in all_results.items():
            vector_score = result.get('vector_score', 0)
            text_score = result.get('text_score', 0)
            # Weighted combination
            combined_score = 0.7 * vector_score + 0.3 * text_score
            result['combined_score'] = combined_score
        
        # Sort by combined score and return top_k
        sorted_results = sorted(all_results.values(), key=lambda x: x['combined_score'], reverse=True)
        return sorted_results[:top_k]
    
    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format search results for consistent output"""
        formatted_results = []
        
        for result in results:
            formatted_result = {
                'id': result.get('id', ''),
                'content': result.get('content', ''),
                'title': result.get('title', ''),
                'document_id': result.get('document_id', ''),
                'chunk_index': result.get('chunk_index', 0),
                'source': result.get('source', ''),
                'category': result.get('category', ''),
                'score': result.get('similarity', result.get('combined_score', result.get('score', 0))),
                'metadata': result.get('metadata', {})
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document and all its chunks"""
        try:
            return self.vector_db.delete_document_chunks(doc_id)
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents indexed"""
        try:
            return self.vector_db.get_document_count()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store"""
        try:
            # Test database connectivity
            doc_count = self.vector_db.get_document_count()
            
            return {
                'status': 'healthy',
                'approach': 'custom',
                'database_name': settings.cosmos_database_name,
                'container_name': settings.cosmos_container_name,
                'document_count': doc_count
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'approach': 'custom',
                'error': str(e)
            }