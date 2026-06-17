import asyncio
import os
import pathlib

# This example connects to one configured Neo4j instance. Cognee's backend
# access-control mode expects the Neo4j Aura provisioning handler instead, so
# keep it disabled here unless the caller explicitly exported another value.
# Set these before Cognee config is initialized so the example uses the intended values.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

import cognee
from cognee import SearchType


async def main():
    """
    Example script demonstrating how to use Cognee with Neo4j

    This example:
    1. Configures Cognee to use Neo4j as graph database
    2. Sets up data directories
    3. Stores sample data with remember to Cognee
    4. Performs different types of searches
    """

    # Set up Neo4j credentials in .env file and get the values from environment variables.
    neo4j_url = os.getenv("GRAPH_DATABASE_URL") or os.getenv("NEO4J_URL") or "bolt://localhost:7687"
    neo4j_user = os.getenv("GRAPH_DATABASE_USERNAME") or os.getenv("NEO4J_USERNAME") or "neo4j"
    neo4j_pass = os.getenv("GRAPH_DATABASE_PASSWORD") or os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("GRAPH_DATABASE_NAME") or os.getenv("NEO4J_DATABASE") or "neo4j"

    if not neo4j_pass:
        raise EnvironmentError(
            "Missing Neo4j password. Set GRAPH_DATABASE_PASSWORD or NEO4J_PASSWORD."
        )

    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "lancedb",
            "vector_dataset_database_handler": "lancedb",
        }
    )

    # Configure Neo4j as the graph database provider
    cognee.config.set_graph_db_config(
        {
            "graph_database_url": neo4j_url,  # Neo4j Bolt URL
            "graph_database_name": neo4j_database,
            "graph_database_provider": "neo4j",  # Specify Neo4j as provider
            "graph_database_username": neo4j_user,  # Neo4j username
            "graph_database_password": neo4j_pass,  # Neo4j password
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
    # await cognee.forget(everything=True)

    # Create a dataset
    dataset_name = "neo4j_example"

    # Add sample text to the dataset
    sample_text = (
        "Neo4j is a graph database management system. "
        "It stores data in nodes and relationships rather than tables as in traditional "
        "relational databases. "
        "Neo4j provides a powerful query language called Cypher for graph traversal and "
        "analysis. "
        "It now supports vector indexing for similarity search with the vector index plugin. "
        "Neo4j allows embedding generation and vector search to be combined with graph "
        "operations. "
        "Applications can use Neo4j to connect vector search with graph context for more "
        "meaningful results."
    )

    # Add the sample text to the dataset
    await cognee.remember([sample_text], dataset_name=dataset_name, self_improvement=False)

    # Now let's perform some searches
    # 1. Search for insights related to "Neo4j"
    insights_results = await cognee.recall(
        query_type=SearchType.GRAPH_COMPLETION, query_text="Neo4j"
    )
    print("\nInsights about Neo4j:")
    for result in insights_results:
        print(f"- {result}")

    # 2. Search for text chunks related to "graph database"
    chunks_results = await cognee.recall(
        query_type=SearchType.CHUNKS, query_text="graph database", datasets=[dataset_name]
    )
    print("\nChunks about graph database:")
    for result in chunks_results:
        print(f"- {result}")

    # 3. Get graph completion related to databases
    graph_completion_results = await cognee.recall(
        query_type=SearchType.GRAPH_COMPLETION, query_text="database"
    )
    print("\nGraph completion for databases:")
    for result in graph_completion_results:
        print(f"- {result}")

    # Clean up (optional)
    # await cognee.forget(everything=True)


if __name__ == "__main__":
    asyncio.run(main())
