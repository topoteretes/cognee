import os
import pathlib
import asyncio
import cognee
from cognee.modules.search.types import SearchType

async def main():
    """
    Example script demonstrating how to use Cognee with Weaviate
    
    This example:
    1. Configures Cognee to use Weaviate as vector database
    2. Sets up data directories
    3. Adds sample data to Cognee
    4. Processes (cognifies) the data
    5. Performs different types of searches
    """
    # Configure Weaviate as the vector database provider

    cognee.config.set_vector_db_config({
        "vector_db_url": "",                      # Set your Weaviate Endpoint
        "vector_db_key": "",                      # Set your Weaviate API key 
        "vector_db_provider": "weaviate",         # Specify Weaviate as provider
    })

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
    dataset_name = "weaviate_example"

    # Add sample text to the dataset
    sample_text = """Weaviate is an open-source vector database that stores both objects and vectors.
    It enables vector search with GraphQL-based filtering capabilities.
    Weaviate can be deployed in the cloud, on-premise, or embedded in your application.
    It allows users to search through vectors using different algorithms and metrics.
    Weaviate supports various modules for text2vec transformations, including BERT, OpenAI, and other models.
    It can index data in multiple ways and offers features like semantic search, classification, and contextualization."""

    # Add the sample text to the dataset
    await cognee.add([sample_text], dataset_name)

    # Process the added document to extract knowledge
    await cognee.cognify([dataset_name])

    # Now let's perform some searches
    # 1. Search for insights related to "Weaviate"
    insights_results = await cognee.search(
        query_type=SearchType.INSIGHTS,
        query_text="Weaviate"
    )
    print("\nInsights about Weaviate:")
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