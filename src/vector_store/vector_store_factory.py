from typing import Union, List
from .base_vector_store import VectorStoreInterface
from .integrated_vector_store import IntegratedVectorStore
from .custom_vector_store import CustomVectorStore
from ..config.settings import settings, VectorizationApproach
import logging

logger = logging.getLogger(__name__)

class VectorStoreFactory:
    """Factory to create appropriate vector store based on configuration"""
    
    @staticmethod
    def create_vector_store() -> VectorStoreInterface:
        """Create vector store based on configuration"""
        try:
            if settings.vectorization_approach == VectorizationApproach.INTEGRATED:
                logger.info("Creating integrated vector store (Azure AI Search)")
                return IntegratedVectorStore()
            elif settings.vectorization_approach == VectorizationApproach.CUSTOM:
                logger.info("Creating custom vector store (separate Azure services)")
                return CustomVectorStore()
            else:
                raise ValueError(f"Unknown vectorization approach: {settings.vectorization_approach}")
                
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    @staticmethod
    def get_available_approaches() -> List[str]:
        """Get list of available vectorization approaches"""
        return [approach.value for approach in VectorizationApproach]
    
    @staticmethod
    def validate_approach(approach: str) -> bool:
        """Validate if approach is supported"""
        return approach in VectorStoreFactory.get_available_approaches()