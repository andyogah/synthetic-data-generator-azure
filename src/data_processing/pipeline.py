from typing import List, Dict, Any, Optional
from ..vector_store.vector_store_factory import VectorStoreFactory
from ..vector_store.base_vector_store import SearchType
from ..config.settings import settings, VectorizationApproach
import logging

logger = logging.getLogger(__name__)

class DataProcessingPipeline:
    """Main pipeline supporting both integrated and custom vectorization approaches"""
    
    def __init__(self, approach: Optional[str] = None):
        """Initialize pipeline with specified approach or use default from settings"""
        if approach:
            if not VectorStoreFactory.validate_approach(approach):
                raise ValueError(f"Invalid approach: {approach}")
            # Temporarily override settings
            original_approach = settings.vectorization_approach
            settings.vectorization_approach = VectorizationApproach(approach)
            self.vector_store = VectorStoreFactory.create_vector_store()
            settings.vectorization_approach = original_approach
        else:
            self.vector_store = VectorStoreFactory.create_vector_store()
        
        self.current_approach = settings.vectorization_approach
        logger.info(f"Initialized pipeline with {self.current_approach} vectorization approach")
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process documents using the configured approach"""
        try:
            logger.info(f"Processing {len(documents)} documents with {self.current_approach} approach")
            
            # Validate documents
            validated_docs = self._validate_documents(documents)
            
            # Index documents
            results = self.vector_store.index_documents(validated_docs)
            
            logger.info(f"Successfully processed {results.get('success_count', 0)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise
    
    def search_documents(self, query: str, search_type: str = "hybrid", 
                        top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents using the configured approach"""
        try:
            # Convert string to SearchType enum
            search_type_enum = SearchType(search_type.lower())
            
            logger.info(f"Searching with query: '{query}' using {search_type} search")
            
            results = self.vector_store.search(
                query=query,
                search_type=search_type_enum,
                top_k=top_k,
                filters=filters
            )
            
            logger.info(f"Search completed, found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document using the configured approach"""
        try:
            success = self.vector_store.delete_document(doc_id)
            if success:
                logger.info(f"Document {doc_id} deleted successfully")
            else:
                logger.warning(f"Failed to delete document {doc_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise
    
    def switch_approach(self, new_approach: str) -> bool:
        """Switch between custom and integrated approaches"""
        try:
            if not VectorStoreFactory.validate_approach(new_approach):
                raise ValueError(f"Invalid approach: {new_approach}")
            
            logger.info(f"Switching from {self.current_approach} to {new_approach}")
            
            # Update settings
            settings.vectorization_approach = VectorizationApproach(new_approach)
            
            # Create new vector store
            self.vector_store = VectorStoreFactory.create_vector_store()
            self.current_approach = new_approach
            
            logger.info(f"Successfully switched to {new_approach} approach")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch approach: {e}")
            return False
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline configuration"""
        health_check = self.vector_store.health_check()
        
        return {
            'current_approach': self.current_approach,
            'available_approaches': VectorStoreFactory.get_available_approaches(),
            'document_count': self.vector_store.get_document_count(),
            'health_status': health_check,
            'search_types': [st.value for st in SearchType],
            'settings': {
                'chunk_size': settings.chunk_size,
                'chunk_overlap': settings.chunk_overlap,
                'embedding_dimension': settings.embedding_dimension,
                'max_search_results': settings.max_search_results
            }
        }
    
    def _validate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and prepare documents for processing"""
        validated_docs = []
        
        for doc in documents:
            # Ensure required fields
            if 'id' not in doc:
                raise ValueError("Document must have 'id' field")
            if 'content' not in doc:
                raise ValueError("Document must have 'content' field")
            
            # Set default values
            validated_doc = {
                'id': doc['id'],
                'content': doc['content'],
                'title': doc.get('title', f"Document {doc['id']}"),
                'source': doc.get('source', 'unknown'),
                'category': doc.get('category', 'general'),
                'metadata': doc.get('metadata', {})
            }
            
            validated_docs.append(validated_doc)
        
        return validated_docs
    
    def batch_process_documents(self, documents: List[Dict[str, Any]], 
                              batch_size: int = 10) -> Dict[str, Any]:
        """Process documents in batches for better performance"""
        try:
            total_docs = len(documents)
            batches = [documents[i:i + batch_size] for i in range(0, total_docs, batch_size)]
            
            total_results = {
                'success_count': 0,
                'failed_count': 0,
                'total_processed': 0,
                'approach': self.current_approach,
                'batches_processed': 0,
                'errors': []
            }
            
            for i, batch in enumerate(batches):
                try:
                    logger.info(f"Processing batch {i + 1}/{len(batches)} with {len(batch)} documents")
                    
                    batch_results = self.process_documents(batch)
                    
                    # Aggregate results
                    total_results['success_count'] += batch_results.get('success_count', 0)
                    total_results['failed_count'] += batch_results.get('failed_count', 0)
                    total_results['total_processed'] += len(batch)
                    total_results['batches_processed'] += 1
                    
                    if 'errors' in batch_results:
                        total_results['errors'].extend(batch_results['errors'])
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {i + 1}: {e}")
                    total_results['failed_count'] += len(batch)
                    total_results['errors'].append(f"Batch {i + 1} failed: {str(e)}")
            
            logger.info(f"Batch processing completed: {total_results['success_count']} successful, "
                       f"{total_results['failed_count']} failed")
            
            return total_results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise