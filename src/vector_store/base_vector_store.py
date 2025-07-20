from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum

class SearchType(str, Enum):
    TEXT = "text"
    VECTOR = "vector"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

class VectorStoreInterface(ABC):
    """Abstract interface for vector stores"""
    
    @abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index documents with vectorization"""
        pass
    
    @abstractmethod
    def search(self, query: str, search_type: SearchType = SearchType.HYBRID, 
               top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents with specified search type"""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete document and all its chunks"""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get total number of documents indexed"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store"""
        pass