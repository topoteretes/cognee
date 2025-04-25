import os
import pathlib
import asyncio
import cognee
from cognee.modules.search.types import SearchType

async def main():
    """
    Example script demonstrating how to use Cognee with Qdrant
    
    This example:
    1. Configures Cognee to use Qdrant as vector database
    2. Sets up data directories
    3. Adds sample data to Cognee
    4. Processes (cognifies) the data
    5. Performs different types of searches
    """
    # Configure Qdrant as the vector database provider
    # Option 1: Simple configuration with just the provider
    cognee.config.set_vector_db_provider("qdrant")
    
    # Option 2: More detailed configuration
    # cognee.config.set_vector_db_config({
    #     "vector_db_url": "http://localhost:6333",  # Default Qdrant URL
    #     "vector_db_key": "",                       # API key if needed
    #     "vector_db_provider": "qdrant",            # Specify Qdrant as provider
    # })

    # Set up data directories for storing documents and system files
    # You should adjust these paths to your needs
    current_dir = pathlib.Path(__file__).parent
    data_directory_path = str(current_dir / "data_storage")
    cognee.config.data_root_directory(data_directory_path)
    
    cognee_directory_path = str(current_dir / "cognee_system")
    cognee.config.system_root_directory(cognee_directory_path)

    # Clean any existing data (optional)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Create a dataset
    dataset_name = "qdrant_example"

    # Add sample text to the dataset
    sample_text = """Qdrant is a vector similarity search engine and vector database.
    It provides a production-ready service with a convenient API for storing, searching, and managing vectors.
    Qdrant supports filtering during vector search, which is essential for real-world applications.
    The database implements various performance optimizations, including HNSW index for approximate nearest neighbor search.
    Qdrant can be deployed via Docker, as a managed cloud service, or directly on bare metal.
    It also supports payload and metadata storage alongside the vectors, allowing for rich data retrieval."""

    # Add the sample text to the dataset
    await cognee.add([sample_text], dataset_name)

    # Process the added document to extract knowledge
    await cognee.cognify([dataset_name])

    # Now let's perform some searches
    # 1. Search for insights related to "Qdrant"
    insights_results = await cognee.search(
        query_type=SearchType.INSIGHTS,
        query_text="Qdrant"
    )
    print("\nInsights about Qdrant:")
    for result in insights_results:
        print(f"- {result}")

    # 2. Search for text chunks related to "vector search"
    chunks_results = await cognee.search(
        query_type=SearchType.CHUNKS,
        query_text="vector search",
        datasets=[dataset_name]
    )
    print("\nChunks about vector search:")
    for result in chunks_results:
        print(f"- {result}")

    # 3. Get graph completion related to databases
    graph_completion_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="database"
    )
    print("\nGraph completion for databases:")
    for result in graph_completion_results:
        print(f"- {result}")

    # Clean up (optional)
    # await cognee.prune.prune_data()
    # await cognee.prune.prune_system(metadata=True)

if __name__ == "__main__":
    asyncio.run(main()) 