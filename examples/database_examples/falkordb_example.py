import os
import pathlib
import asyncio
import cognee
from cognee.modules.search.types import SearchType


async def main():
    """
    Example script demonstrating how to use Cognee with FalkorDB

    This example:
    1. Configures Cognee to use FalkorDB as graph database
    2. Sets up data directories
    3. Adds sample data to Cognee
    4. Processes (cognifies) the data
    5. Performs different types of searches
    """
    # Configure FalkorDB as the graph database provider
    cognee.config.set_graph_db_config(
        {
            "graph_database_url": "localhost",  # FalkorDB URL (using Redis protocol)
            "graph_database_port": 6379,
            "graph_database_provider": "falkordb",
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
    dataset_name = "falkordb_example"

    # Add sample text to the dataset
    sample_text = """FalkorDB is a graph database that evolved from RedisGraph.
    It is focused on providing high-performance graph operations.
    FalkorDB uses sparse adjacency matrices to represent the graph data structure.
    It supports the Cypher query language for querying graph data.
    FalkorDB can be integrated with vector search capabilities for AI applications.
    It provides a Redis module, allowing users to leverage Redis's features alongside graph capabilities."""

    # Add the sample text to the dataset
    await cognee.add([sample_text], dataset_name)

    # Process the added document to extract knowledge
    await cognee.cognify([dataset_name])

    # Now let's perform some searches
    # 1. Search for insights related to "FalkorDB"
    insights_results = await cognee.search(query_type=SearchType.INSIGHTS, query_text="FalkorDB")
    print("\nInsights about FalkorDB:")
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
