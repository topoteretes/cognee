import os
import pathlib
import asyncio
import cognee
from cognee.modules.search.types import SearchType

async def main():
    """
    Example script demonstrating how to use Cognee with Milvus
    
    This example:
    1. Configures Cognee to use Milvus as vector database
    2. Sets up data directories
    3. Adds sample data to Cognee
    4. Processes (cognifies) the data
    5. Performs different types of searches
    """
    # Configure Milvus as the vector database provider
    cognee.config.set_vector_db_config({
        "vector_db_url": "",      # Enter Milvus Endpoint
        "vector_db_key": "",                     # Enter Token
        "vector_db_provider": "milvus",          # Specify Milvus as provider
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
    dataset_name = "milvus_example"

    # Add sample text to the dataset
    sample_text = """Milvus is an open-source vector database built to power AI applications.
    It is designed for storing, indexing, and querying large-scale vector datasets.
    Milvus implements efficient approximate nearest neighbor search algorithms.
    It features advanced indexing techniques like HNSW, IVF, PQ, and more.
    Milvus supports hybrid searches combining vector similarity with scalar filtering.
    The system can be deployed standalone, in clusters, or through a cloud service."""

    # Add the sample text to the dataset
    await cognee.add([sample_text], dataset_name)

    # Process the added document to extract knowledge
    await cognee.cognify([dataset_name])

    # Now let's perform some searches
    # 1. Search for insights related to "Milvus"
    insights_results = await cognee.search(
        query_type=SearchType.INSIGHTS,
        query_text="Milvus"
    )
    print("\nInsights about Milvus:")
    for result in insights_results:
        print(f"- {result}")

    # 2. Search for text chunks related to "vector similarity"
    chunks_results = await cognee.search(
        query_type=SearchType.CHUNKS,
        query_text="vector similarity",
        datasets=[dataset_name]
    )
    print("\nChunks about vector similarity:")
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