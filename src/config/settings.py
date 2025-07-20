
from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings


class VectorizationApproach(str, Enum):
    CUSTOM = "custom"
    INTEGRATED = "integrated"

class SearchType(str, Enum):
    TEXT = "text"
    VECTOR = "vector"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

class Settings(BaseSettings):
    # Vectorization approach selection
    vectorization_approach: VectorizationApproach = VectorizationApproach.INTEGRATED
    
    # Azure AI Search (Integrated approach)
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index_name: str = "synthetic-data-index"
    azure_search_api_version: str = "2023-11-01"
    
    # Custom approach settings
    azure_blob_connection_string: str = ""
    azure_cosmos_connection_string: str = ""
    cosmos_database_name: str = "synthetic_data"
    cosmos_container_name: str = "vectors"
    
    # Azure OpenAI (used by both approaches)
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_embedding_model: str = "text-embedding-ada-002"
    azure_openai_chat_model: str = "gpt-4"
    azure_openai_api_version: str = "2023-12-01-preview"
    
    # Processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_dimension: int = 1536
    max_search_results: int = 10
    
    # Search configuration
    default_search_type: SearchType = SearchType.HYBRID
    enable_semantic_search: bool = True
    enable_vector_search: bool = True
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()