from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
import numpy as np
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self, blob_connection_string, cosmos_connection_string, database_name, container_name):
        self.blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        self.cosmos_client = CosmosClient.from_connection_string(cosmos_connection_string)
        self.database_name = database_name
        self.container_name = container_name
        
        # Create database and container if they don't exist
        self.database = self.cosmos_client.create_database_if_not_exists(database_name)
        self.container = self.database.create_container_if_not_exists(
            id=container_name,
            partition_key={'paths': ['/document_id'], 'kind': 'Hash'}
        )
        
        logger.info(f"Connected to Cosmos DB: {database_name}/{container_name}")

    def store_vector_with_metadata(self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        """Store vector with associated metadata"""
        try:
            vector_data = {
                'id': vector_id,
                'vector': vector.tolist(),
                'content': metadata.get('content', ''),
                'title': metadata.get('title', ''),
                'document_id': metadata.get('document_id', ''),
                'chunk_index': metadata.get('chunk_index', 0),
                'source': metadata.get('source', ''),
                'category': metadata.get('category', ''),
                'created_at': metadata.get('created_at', datetime.utcnow().isoformat()),
                'metadata': metadata.get('metadata', {})
            }
            
            self.container.upsert_item(vector_data)
            logger.debug(f"Stored vector with ID: {vector_id}")
            
        except Exception as e:
            logger.error(f"Failed to store vector {vector_id}: {e}")
            raise

    def store_vector(self, vector_id, vector):
        """Legacy method for backward compatibility"""
        try:
            vector_data = {
                'id': vector_id,
                'vector': vector.tolist(),
                'document_id': vector_id,
                'content': '',
                'created_at': datetime.utcnow().isoformat()
            }
            self.container.upsert_item(vector_data)
            
        except Exception as e:
            logger.error(f"Failed to store vector {vector_id}: {e}")
            raise

    def retrieve_vector(self, vector_id):
        """Retrieve vector by ID"""
        try:
            document_id = vector_id.split('_chunk_')[0] if '_chunk_' in vector_id else vector_id
            item = self.container.read_item(item=vector_id, partition_key=document_id)
            return np.array(item['vector'])
            
        except Exception as e:
            logger.error(f"Error retrieving vector {vector_id}: {e}")
            return None

    def search_vectors(self, query_vector: np.ndarray, top_k: int = 5, 
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors using cosine similarity"""
        try:
            # Build query with optional filters
            query_conditions = []
            if filters:
                for key, value in filters.items():
                    if isinstance(value, str):
                        query_conditions.append(f"c.{key} = '{value}'")
                    else:
                        query_conditions.append(f"c.{key} = {value}")
            
            where_clause = " AND ".join(query_conditions) if query_conditions else "1=1"
            query = f"SELECT * FROM c WHERE {where_clause}"
            
            all_vectors = list(self.container.query_items(query, enable_cross_partition_query=True))
            
            similarities = []
            
            for item in all_vectors:
                stored_vector = np.array(item['vector'])
                
                # Calculate cosine similarity
                similarity = np.dot(query_vector, stored_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
                )
                
                result = {
                    'id': item['id'],
                    'similarity': float(similarity),
                    'content': item.get('content', ''),
                    'title': item.get('title', ''),
                    'document_id': item.get('document_id', ''),
                    'chunk_index': item.get('chunk_index', 0),
                    'source': item.get('source', ''),
                    'category': item.get('category', ''),
                    'metadata': item.get('metadata', {})
                }
                similarities.append(result)
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    def text_search(self, query: str, top_k: int = 5, 
                   filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Text-based search using content matching"""
        try:
            # Build query with text search and optional filters
            query_conditions = [f"CONTAINS(c.content, '{query}', true)"]
            
            if filters:
                for key, value in filters.items():
                    if isinstance(value, str):
                        query_conditions.append(f"c.{key} = '{value}'")
                    else:
                        query_conditions.append(f"c.{key} = {value}")
            
            where_clause = " AND ".join(query_conditions)
            sql_query = f"SELECT * FROM c WHERE {where_clause}"
            
            results = list(self.container.query_items(sql_query, enable_cross_partition_query=True))
            
            # Simple scoring based on query term frequency
            scored_results = []
            query_terms = query.lower().split()
            
            for item in results:
                content = item.get('content', '').lower()
                score = sum(content.count(term) for term in query_terms)
                
                result = {
                    'id': item['id'],
                    'score': score,
                    'content': item.get('content', ''),
                    'title': item.get('title', ''),
                    'document_id': item.get('document_id', ''),
                    'chunk_index': item.get('chunk_index', 0),
                    'source': item.get('source', ''),
                    'category': item.get('category', ''),
                    'metadata': item.get('metadata', {})
                }
                scored_results.append(result)
            
            # Sort by score (descending)
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            return scored_results[:top_k]
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    def semantic_search(self, query_vector: np.ndarray, query_text: str, top_k: int = 5,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Enhanced semantic search combining vector similarity and text relevance"""
        try:
            # Get vector search results
            vector_results = self.search_vectors(query_vector, top_k * 2, filters)
            
            # Get text search results
            text_results = self.text_search(query_text, top_k * 2, filters)
            
            # Combine results with weighted scoring
            combined_results = {}
            
            # Add vector results
            for result in vector_results:
                doc_id = result['id']
                combined_results[doc_id] = result
                combined_results[doc_id]['vector_score'] = result['similarity']
                combined_results[doc_id]['text_score'] = 0
            
            # Add text results
            for result in text_results:
                doc_id = result['id']
                if doc_id in combined_results:
                    combined_results[doc_id]['text_score'] = result['score']
                else:
                    combined_results[doc_id] = result
                    combined_results[doc_id]['vector_score'] = 0
                    combined_results[doc_id]['text_score'] = result['score']
            
            # Calculate semantic score
            for doc_id, result in combined_results.items():
                vector_score = result.get('vector_score', 0)
                text_score = result.get('text_score', 0)
                
                # Normalize text score
                max_text_score = max([r.get('text_score', 0) for r in combined_results.values()])
                normalized_text_score = text_score / max_text_score if max_text_score > 0 else 0
                
                # Weighted combination
                semantic_score = 0.8 * vector_score + 0.2 * normalized_text_score
                result['semantic_score'] = semantic_score
            
            # Sort by semantic score
            sorted_results = sorted(combined_results.values(), key=lambda x: x['semantic_score'], reverse=True)
            return sorted_results[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def delete_document_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            query = f"SELECT c.id FROM c WHERE c.document_id = '{document_id}'"
            chunks = list(self.container.query_items(query, enable_cross_partition_query=True))
            
            deleted_count = 0
            for chunk in chunks:
                try:
                    self.container.delete_item(item=chunk['id'], partition_key=document_id)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete chunk {chunk['id']}: {e}")
            
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete document chunks for {document_id}: {e}")
            return False

    def delete_vector(self, vector_id):
        """Delete a specific vector"""
        try:
            document_id = vector_id.split('_chunk_')[0] if '_chunk_' in vector_id else vector_id
            self.container.delete_item(item=vector_id, partition_key=document_id)
            logger.info(f"Deleted vector: {vector_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            raise

    def get_document_count(self) -> int:
        """Get total number of unique documents"""
        try:
            query = "SELECT VALUE COUNT(1) FROM (SELECT DISTINCT c.document_id FROM c)"
            result = list(self.container.query_items(query, enable_cross_partition_query=True))
            return result[0] if result else 0
            
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0