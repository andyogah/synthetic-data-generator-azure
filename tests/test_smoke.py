# Please note: This is a placeholder smoke test.
# The test suite will be completed to include:
# - Unit tests for each module (loader, preprocessor, chunker, embedder, vector store, etc.)
# - Integration tests for the full pipeline
# - Edge case and error handling tests
# - Mocking of external Azure services
# - Validation of expected outputs

from src.data_processing.pipeline import DataProcessingPipeline
from src.config.settings import Settings

def test_pipeline_runs():
    settings = Settings()
    pipeline = DataProcessingPipeline(settings)
    # This should not raise exceptions; adjust as needed for your pipeline
    pipeline.get_pipeline_info()  # Replace with the actual method name from your pipeline class
    
