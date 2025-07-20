from src.data_processing.pipeline import DataProcessingPipeline
from src.config.settings import settings
from src.vector_store.base_vector_store import SearchType
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main application demonstrating both approaches"""
    
    # Initialize pipeline with default approach from settings
    pipeline = DataProcessingPipeline()
    
    # Display pipeline information
    pipeline_info = pipeline.get_pipeline_info()
    logger.info(f"Pipeline Info: {json.dumps(pipeline_info, indent=2)}")
    
    # Sample documents
    documents = [
        {
            'id': 'doc1',
            'content': 'This is sample content for document 1 about artificial intelligence and machine learning.',
            'title': 'AI and ML Document',
            'source': 'research_paper',
            'category': 'technology',
            'metadata': {'author': 'John Doe', 'year': 2023}
        },
        {
            'id': 'doc2',
            'content': 'This document discusses natural language processing and deep learning techniques.',
            'title': 'NLP and Deep Learning',
            'source': 'research_paper',
            'category': 'technology',
            'metadata': {'author': 'Jane Smith', 'year': 2023}
        },
        {
            'id': 'doc3',
            'content': 'Azure services provide comprehensive cloud computing solutions for businesses.',
            'title': 'Azure Cloud Services',
            'source': 'documentation',
            'category': 'cloud',
            'metadata': {'version': '2023.1', 'type': 'guide'}
        }
    ]
    
    try:
        # Process documents
        logger.info("=== Processing Documents ===")
        results = pipeline.process_documents(documents)
        logger.info(f"Processing Results: {json.dumps(results, indent=2)}")
        
        # Demonstrate different search types
        search_queries = [
            ("machine learning", "vector"),
            ("artificial intelligence", "text"),
            ("cloud computing", "semantic"),
            ("Azure AI", "hybrid")
        ]
        
        for query, search_type in search_queries:
            logger.info(f"\n=== {search_type.upper()} Search: '{query}' ===")
            search_results = pipeline.search_documents(
                query=query,
                search_type=search_type,
                top_k=3
            )
            
            for i, result in enumerate(search_results, 1):
                logger.info(f"Result {i}:")
                logger.info(f"  Title: {result.get('title', 'N/A')}")
                logger.info(f"  Content: {result.get('content', 'N/A')[:100]}...")
                logger.info(f"  Score: {result.get('score', 0):.4f}")
                logger.info(f"  Source: {result.get('source', 'N/A')}")
        
        # Demonstrate approach switching
        logger.info("\n=== Switching Approaches ===")
        current_approach = pipeline.current_approach
        
        # Get available approaches
        available_approaches = pipeline.get_pipeline_info()['available_approaches']
        other_approach = next(
            (approach for approach in available_approaches if approach != current_approach),
            None
        )
        
        if other_approach:
            logger.info(f"Switching from {current_approach} to {other_approach}")
            
            if pipeline.switch_approach(other_approach):
                logger.info(f"Successfully switched to {other_approach}")
                
                # Test search with new approach
                search_results = pipeline.search_documents(
                    query="machine learning",
                    search_type="hybrid",
                    top_k=2
                )
                
                logger.info(f"Search with {other_approach} approach returned {len(search_results)} results")
                
                # Switch back
                pipeline.switch_approach(current_approach)
                logger.info(f"Switched back to {current_approach}")
            else:
                logger.error(f"Failed to switch to {other_approach}")
        
        # Demonstrate batch processing
        logger.info("\n=== Batch Processing ===")
        batch_results = pipeline.batch_process_documents(documents, batch_size=2)
        logger.info(f"Batch Processing Results: {json.dumps(batch_results, indent=2)}")
        
        # Final pipeline info
        final_info = pipeline.get_pipeline_info()
        logger.info(f"\nFinal Pipeline Info: {json.dumps(final_info, indent=2)}")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

def demonstrate_approach_comparison():
    """Demonstrate the differences between integrated and custom approaches"""
    
    logger.info("\n=== APPROACH COMPARISON ===")
    
    # Sample document for comparison
    test_doc = {
        'id': 'comparison_test',
        'content': 'This is a test document for comparing integrated and custom vectorization approaches.',
        'title': 'Comparison Test',
        'source': 'test',
        'category': 'comparison'
    }
    
    approaches = ['integrated', 'custom']
    
    for approach in approaches:
        try:
            logger.info(f"\n--- Testing {approach.upper()} Approach ---")
            
            # Create pipeline with specific approach
            pipeline = DataProcessingPipeline(approach=approach)
            
            # Process document
            process_results = pipeline.process_documents([test_doc])
            logger.info(f"Processing: {process_results}")
            
            # Search document
            search_results = pipeline.search_documents(
                query="vectorization approaches",
                search_type="hybrid",
                top_k=1
            )
            logger.info(f"Search: Found {len(search_results)} results")
            
            # Get health check
            health = pipeline.vector_store.health_check()
            logger.info(f"Health: {health}")
            
        except Exception as e:
            logger.error(f"Error with {approach} approach: {e}")

if __name__ == "__main__":
    # Run main application
    main()
    
    # Run approach comparison
    demonstrate_approach_comparison()