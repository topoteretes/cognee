import os
import pathlib
import asyncio
import cognee
from cognee.modules.search.types import SearchType


async def main():
    """
    Example script demonstrating how to use Cognee with ChromaDB

    This example:
    1. Configures Cognee to use ChromaDB as vector database
    2. Sets up data directories
    3. Adds sample data to Cognee
    4. Processes (cognifies) the data
    5. Performs different types of searches
    """
    # Configure ChromaDB as the vector database provider
    cognee.config.set_vector_db_config(
        {
            "vector_db_url": "http://localhost:8000",  # Default ChromaDB server URL
            "vector_db_key": "",  # ChromaDB doesn't require an API key by default
            "vector_db_provider": "chromadb",  # Specify ChromaDB as provider
        }
    )

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
    dataset_name = "chromadb_example"

    # Add sample text to the dataset
    sample_text = """ChromaDB is an open-source embedding database.
    It allows users to store and query embeddings and their associated metadata.
    ChromaDB can be deployed in various ways: in-memory, on disk via sqlite, or as a persistent service.
    It is designed to be fast, scalable, and easy to use, making it a popular choice for AI applications.
    The database is built to handle vector search efficiently, which is essential for semantic search applications.
    ChromaDB supports multiple distance metrics for vector similarity search and can be integrated with various ML frameworks."""

    # Add the sample text to the dataset
    await cognee.add([sample_text], dataset_name)

    # Process the added document to extract knowledge
    await cognee.cognify([dataset_name])

    # Now let's perform some searches
    # 1. Search for insights related to "ChromaDB"
    insights_results = await cognee.search(query_type=SearchType.INSIGHTS, query_text="ChromaDB")
    print("\nInsights about ChromaDB:")
    for result in insights_results:
        print(f"- {result}")

    # 2. Search for text chunks related to "vector search"
    chunks_results = await cognee.search(
        query_type=SearchType.CHUNKS, query_text="vector search", datasets=[dataset_name]
    )
    print("\nChunks about vector search:")
    for result in chunks_results:
        print(f"- {result}")

    # 3. Get graph completion related to databases
    graph_completion_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text="database"
    )
    print("\nGraph completion for databases:")
    for result in graph_completion_results:
        print(f"- {result}")

    # Clean up (optional)
    # await cognee.prune.prune_data()
    # await cognee.prune.prune_system(metadata=True)


if __name__ == "__main__":
    asyncio.run(main())
