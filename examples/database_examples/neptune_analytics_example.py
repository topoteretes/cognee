import base64
import json
import os
import pathlib
import asyncio
import cognee
from cognee.modules.search.types import SearchType
from dotenv import load_dotenv

load_dotenv()


async def main():
    """
    Example script demonstrating how to use Cognee with Amazon Neptune Analytics

    This example:
    1. Configures Cognee to use Neptune Analytics as graph database
    2. Sets up data directories
    3. Adds sample data to Cognee
    4. Processes/cognifies the data
    5. Performs different types of searches
    """

    # Set up Amazon credentials in .env file and get the values from environment variables
    graph_endpoint_url = "neptune-graph://" + os.getenv("GRAPH_ID", "")

    # Configure Neptune Analytics as the graph & vector database provider
    cognee.config.set_graph_db_config(
        {
            "graph_database_provider": "neptune_analytics",  # Specify Neptune Analytics as provider
            "graph_database_url": graph_endpoint_url,  # Neptune Analytics endpoint with the format neptune-graph://<GRAPH_ID>
        }
    )
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "neptune_analytics",  # Specify Neptune Analytics as provider
            "vector_db_url": graph_endpoint_url,  # Neptune Analytics endpoint with the format neptune-graph://<GRAPH_ID>
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
    dataset_name = "neptune_example"

    # Add sample text to the dataset
    sample_text_1 = """Neptune Analytics is a memory-optimized graph database engine for analytics. With Neptune
    Analytics, you can get insights and find trends by processing large amounts of graph data in seconds. To analyze
    graph data quickly and easily, Neptune Analytics stores large graph datasets in memory. It supports a library of
    optimized graph analytic algorithms, low-latency graph queries, and vector search capabilities within graph
    traversals.
    """

    sample_text_2 = """Neptune Analytics is an ideal choice for investigatory, exploratory, or data-science workloads 
    that require fast iteration for data, analytical and algorithmic processing, or vector search on graph data. It 
    complements Amazon Neptune Database, a popular managed graph database. To perform intensive analysis, you can load 
    the data from a Neptune Database graph or snapshot into Neptune Analytics. You can also load graph data that's 
    stored in Amazon S3.
    """

    # Add the sample text to the dataset
    await cognee.add([sample_text_1, sample_text_2], dataset_name)

    # Process the added document to extract knowledge
    await cognee.cognify([dataset_name])

    # Now let's perform some searches
    # 1. Search for insights related to "Neptune Analytics"
    insights_results = await cognee.search(
        query_type=SearchType.INSIGHTS, query_text="Neptune Analytics"
    )
    print("\n========Insights about Neptune Analytics========:")
    for result in insights_results:
        print(f"- {result}")

    # 2. Search for text chunks related to "graph database"
    chunks_results = await cognee.search(
        query_type=SearchType.CHUNKS, query_text="graph database", datasets=[dataset_name]
    )
    print("\n========Chunks about graph database========:")
    for result in chunks_results:
        print(f"- {result}")

    # 3. Get graph completion related to databases
    graph_completion_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text="database"
    )
    print("\n========Graph completion for databases========:")
    for result in graph_completion_results:
        print(f"- {result}")

    # Clean up (optional)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


if __name__ == "__main__":
    asyncio.run(main())
