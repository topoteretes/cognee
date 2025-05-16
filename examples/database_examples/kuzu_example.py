import os
import pathlib
import asyncio
import cognee
from cognee.modules.search.types import SearchType


async def main():
    """
    Example script demonstrating how to use Cognee with KuzuDB

    This example:
    1. Configures Cognee to use KuzuDB as graph database
    2. Sets up data directories
    3. Adds sample data to Cognee
    4. Processes (cognifies) the data
    5. Performs different types of searches
    """
    # Configure KuzuDB as the graph database provider
    cognee.config.set_graph_db_config(
        {
            "graph_database_provider": "kuzu",  # Specify KuzuDB as provider
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
    dataset_name = "kuzu_example"

    # Add sample text to the dataset
    sample_text = """KuzuDB is a graph database system optimized for running complex graph analytics.
    It is designed to be a high-performance graph database for data science workloads.
    KuzuDB is built with modern hardware optimizations in mind.
    It provides support for property graphs and offers a Cypher-like query language.
    KuzuDB can handle both transactional and analytical graph workloads.
    The database now includes vector search capabilities for AI applications and semantic search."""

    # Add the sample text to the dataset
    await cognee.add([sample_text], dataset_name)

    # Process the added document to extract knowledge
    await cognee.cognify([dataset_name])

    # Now let's perform some searches
    # 1. Search for insights related to "KuzuDB"
    insights_results = await cognee.search(query_type=SearchType.INSIGHTS, query_text="KuzuDB")
    print("\nInsights about KuzuDB:")
    for result in insights_results:
        print(f"- {result}")

    # 2. Search for text chunks related to "graph database"
    chunks_results = await cognee.search(
        query_type=SearchType.CHUNKS, query_text="graph database", datasets=[dataset_name]
    )
    print("\nChunks about graph database:")
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
