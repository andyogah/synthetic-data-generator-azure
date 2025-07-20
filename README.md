# Synthetic Data Generator

## Overview
The Synthetic Data Generator is a modular Python project for generating synthetic data that closely resembles original datasets. Unlike LangChain-based solutions, this project implements its own data processing pipeline, embedding, and vectorization logic, providing direct integration with Azure services and full control over each step.

### How This Differs from LangChain-Based Generators

- **No LangChain Dependency:** All data loading, chunking, embedding, and vector storage are implemented natively, not via LangChain abstractions.
- **Direct Azure Integration:** Embedding, search, and storage are handled directly with Azure OpenAI, Azure Cognitive Search, and Azure Cosmos DB APIs.
- **Customizable Pipeline:** The orchestration and workflow are fully modular and extensible, allowing for custom logic at each stage.
- **Flexible Vectorization:** Supports both integrated (native Azure Cognitive Search) and custom (OpenAI + Cosmos DB) vectorization approaches, selectable via `.env`.
- **PII Protection:** While LangChain-based solutions may use Presidio, this project is designed for extensibility and can integrate custom PII protection as needed.

## The Challenge: The Data Access Paradox

In high-security organizations, developers often cannot access real data due to privacy, security, and clearance restrictions. This project enables rapid development of POCs and MVPs by generating realistic, contextually-aware synthetic datasets that preserve the statistical and semantic properties of original data, without exposing sensitive information.

## Key Value Propositions

- **Accelerate Development:** Build and test solutions without waiting for data access approvals.
- **Maintain Privacy:** Generate synthetic data with no actual PII or sensitive information.
- **Preserve Context:** Maintain semantic relationships and statistical properties.
- **Enable Innovation:** Work with realistic data regardless of clearance level.
- **Compliance Ready:** Designed for extensibility to support regulatory compliance.

## Features
- **Flexible Data Loading**: Load data from local files, Azure Blob Storage, and other sources.
- **Data Preprocessing**: Clean and normalize data for downstream processing.
- **Chunking**: Split data into manageable, context-preserving chunks.
- **Embedding**: Convert text chunks into vector embeddings using Azure OpenAI or other models.
- **Vector Storage**: Store embeddings in integrated or custom vector databases (Azure Cognitive Search, Cosmos DB, etc.).
- **Synthetic Generation**: Generate synthetic content using LLMs via Azure OpenAI Service.
- **Search Capabilities**: Support for text search, vector search, semantic search, and hybrid search with reranking.
- **Extensible Pipeline**: Modular pipeline for custom workflows and easy integration of new components.
- **Centralized Configuration**: Manage settings and secrets via environment variables and config files.

## Azure Products Used
- **Azure Blob Storage**: Store and retrieve original data files.
- **Azure Cognitive Search**: Text and vector search capabilities.
- **Azure OpenAI Service**: Generate synthetic data using LLMs and embeddings.
- **Azure Cosmos DB**: Store vector embeddings and metadata.

## Project Structure
```
synthetic-data-generator3/
├── src/
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   └── vector_store/
│       ├── __init__.py
│       ├── base_vector_store.py
│       ├── integrated_vector_store.py
│       ├── custom_vector_store.py
│       ├── vector_store_factory.py
│       └── vector_db.py
├── .env
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd synthetic-data-generator3
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file in the root directory:
   ```
   AZURE_API_KEY=<your_azure_api_key>
   AZURE_SEARCH_SERVICE=<your_search_service_name>
   AZURE_OPENAI_API_KEY=<your_openai_api_key>
   AZURE_BLOB_CONNECTION_STRING=<your_blob_connection_string>
   AZURE_COSMOS_CONNECTION_STRING=<your_cosmos_connection_string>
   ```

## Usage

1. **Run the main application:**
   ```
   python src/main.py
   ```

2. **Typical workflow:**
   - Load data using the `Loader` (from file or Azure Blob).
   - Preprocess data with the `Preprocessor`.
   - Chunk data using the `Chunker`.
   - Embed chunks via the `Embedder`.
   - Store embeddings in a vector store (choose via `VectorStoreFactory`).
   - Generate synthetic data with integrated LLMs.
   - Search and retrieve data using text/vector/hybrid search.

### Example: Data Processing Pipeline

```python
from data_processing.pipeline import DataProcessingPipeline
from config.settings import Settings

settings = Settings()
pipeline = DataProcessingPipeline(settings)
pipeline.run()
```

### Example: Custom Vector Store

```python
from vector_store.vector_store_factory import VectorStoreFactory

vector_store = VectorStoreFactory.create('custom', config=settings)
vector_store.store_embedding(embedding, metadata)
```

## Vectorization Approaches

This project supports two distinct vectorization strategies, selectable via the `VECTORIZATION_APPROACH` variable in your `.env` file:

- **Integrated Vectorization (Azure Cognitive Search native):**
  - Set `VECTORIZATION_APPROACH=integrated` in `.env`.
  - Uses Azure Cognitive Search's built-in vectorization and indexing capabilities.
  - Embeddings are stored and searched natively within Azure Cognitive Search.
  - Requires configuration of Azure Search endpoint, API key, index name, and API version.

- **Custom Vectorization:**
  - Set `VECTORIZATION_APPROACH=custom` in `.env`.
  - Embeddings are generated using Azure OpenAI and stored in Azure Cosmos DB.
  - Data files are managed via Azure Blob Storage.
  - Allows for custom logic in embedding generation, storage, and retrieval.
  - Requires configuration of Azure Blob and Cosmos DB connection strings, database, and container names.

Both approaches use Azure OpenAI for embedding and synthetic data generation. The pipeline and vector store modules automatically select the appropriate implementation based on your `.env` settings.

### Example `.env` configuration

```properties
# Vectorization approach selection (custom or integrated)
VECTORIZATION_APPROACH=integrated

# Azure AI Search (Integrated approach)
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your_search_api_key
AZURE_SEARCH_INDEX_NAME=synthetic-data-index
AZURE_SEARCH_API_VERSION=2023-11-01

# Custom approach settings
AZURE_BLOB_CONNECTION_STRING=your_blob_connection_string
AZURE_COSMOS_CONNECTION_STRING=your_cosmos_connection_string
COSMOS_DATABASE_NAME=synthetic_data
COSMOS_CONTAINER_NAME=vectors
```

## Core Components

### Data Processing (`src/data_processing/`)
- **loader.py**: Load data from files, Azure Blob, or other sources.
- **preprocessor.py**: Clean, normalize, and prepare data.
- **chunker.py**: Split data into context-preserving chunks.
- **embedder.py**: Generate vector embeddings using Azure OpenAI.
- **pipeline.py**: Orchestrate the end-to-end data processing workflow.

### Vector Store (`src/vector_store/`)
- **base_vector_store.py**: Abstract base class for vector storage.
- **integrated_vector_store.py**: Azure Cognitive Search integration.
- **custom_vector_store.py**: Custom vector storage implementation (Cosmos DB).
- **vector_store_factory.py**: Factory for creating vector store instances.
- **vector_db.py**: Database operations for vector storage.

### Configuration (`src/config/`)
- **settings.py**: Centralized configuration management, loads environment variables.

### Main Application (`src/main.py`)
- Entry point for running the pipeline and managing workflow.

## Extending the Project

- **Add new data sources**: Implement additional loaders in `data_processing/loader.py`.
- **Custom chunking strategies**: Extend `chunker.py` for domain-specific chunking.
- **New embedding models**: Integrate other embedding providers in `embedder.py`.
- **Additional vector stores**: Implement new stores in `vector_store/` and register with `vector_store_factory.py`.
- **Pipeline customization**: Modify or extend `pipeline.py` for custom workflows.
- **PII Protection**: Integrate custom PII detection and anonymization as needed.

## Troubleshooting

- **Missing environment variables**: Ensure `.env` is correctly configured.
- **Azure authentication errors**: Verify API keys and connection strings.
- **Dependency issues**: Check `requirements.txt` and reinstall packages.
- **Module import errors**: Confirm project structure and PYTHONPATH.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for enhancements or bug fixes. Follow PEP8 and include tests for new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## References

- [synthetic-data-generator-langchain (for comparison)](https://github.com/andyogah/synthetic-data-generator-langchain)
- [Azure Cognitive Search Documentation](https://learn.microsoft.com/en-us/azure/search/)
- [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure Cosmos DB Documentation](https://learn.microsoft.com/en-us/azure/cosmos-db/)