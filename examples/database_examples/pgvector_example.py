import os
import pathlib
import asyncio
import cognee
from cognee.modules.search.types import SearchType


async def main():
    """
    Example script demonstrating how to use Cognee with PGVector

    This example:
    1. Configures Cognee to use PostgreSQL with PGVector extension as vector database
    2. Sets up data directories
    3. Adds sample data to Cognee
    4. Processes (cognifies) the data
    5. Performs different types of searches
    """
    # Configure PGVector as the vector database provider
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "pgvector",  # Specify PGVector as provider
        }
    )

    # Configure PostgreSQL connection details
    # These settings are required for PGVector
    cognee.config.set_relational_db_config(
        {
            "db_path": "",
            "db_name": "cognee_db",
            "db_host": "127.0.0.1",
            "db_port": "5432",
            "db_username": "cognee",
            "db_password": "cognee",
            "db_provider": "postgres",
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
    dataset_name = "pgvector_example"

    # Add sample text to the dataset
    sample_text = """PGVector is an extension for PostgreSQL that adds vector similarity search capabilities.
    It supports multiple indexing methods, including IVFFlat, HNSW, and brute-force search.
    PGVector allows you to store vector embeddings directly in your PostgreSQL database.
    It provides distance functions like L2 distance, inner product, and cosine distance.
    Using PGVector, you can perform both metadata filtering and vector similarity search in a single query.
    The extension is often used for applications like semantic search, recommendations, and image similarity."""

    # Add the sample text to the dataset
    await cognee.add([sample_text], dataset_name)

    # Process the added document to extract knowledge
    await cognee.cognify([dataset_name])

    # Now let's perform some searches
    # 1. Search for insights related to "PGVector"
    insights_results = await cognee.search(query_type=SearchType.INSIGHTS, query_text="PGVector")
    print("\nInsights about PGVector:")
    for result in insights_results:
        print(f"- {result}")

    # 2. Search for text chunks related to "vector similarity"
    chunks_results = await cognee.search(
        query_type=SearchType.CHUNKS, query_text="vector similarity", datasets=[dataset_name]
    )
    print("\nChunks about vector similarity:")
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
