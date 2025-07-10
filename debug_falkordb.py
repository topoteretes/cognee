import os
import cognee
import pathlib
import asyncio


async def debug_falkordb():
    """Debug script to see what's actually stored in FalkorDB"""

    # Check if FalkorDB is available
    try:
        from falkordb import FalkorDB

        client = FalkorDB(host="localhost", port=6379)
        client.list_graphs()
        print("✅ FalkorDB connection successful")
    except Exception as e:
        print(f"❌ FalkorDB not available: {e}")
        return

    # Configure FalkorDB
    cognee.config.set_graph_db_config(
        {
            "graph_database_url": "localhost",
            "graph_database_port": 6379,
            "graph_database_provider": "falkordb",
        }
    )

    cognee.config.set_vector_db_config(
        {
            "vector_db_url": "localhost",
            "vector_db_port": 6379,
            "vector_db_provider": "falkordb",
        }
    )

    # Set up directories
    data_directory_path = str(pathlib.Path("./debug_data").resolve())
    cognee_directory_path = str(pathlib.Path("./debug_cognee").resolve())

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    # Clean up first
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Add simple text
    simple_text = "Artificial Intelligence (AI) is a fascinating technology."
    dataset_name = "test_dataset"

    print("📝 Adding data...")
    await cognee.add([simple_text], dataset_name)

    print("🧠 Running cognify...")
    await cognee.cognify([dataset_name])

    # Debug: Check what's in the database
    print("\n🔍 Checking what's in the database...")

    from cognee.infrastructure.databases.vector import get_vector_engine
    from cognee.infrastructure.databases.graph import get_graph_engine

    vector_engine = get_vector_engine()
    graph_engine = await get_graph_engine()

    # Get all graph data
    print("\n📊 Graph data:")
    graph_data = await graph_engine.get_graph_data()
    nodes, edges = graph_data

    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")

    if nodes:
        print("\n🏷️ Sample nodes:")
        for i, (node_id, node_props) in enumerate(nodes[:3]):
            print(f"  Node {i + 1}: ID={node_id}")
            print(f"    Properties: {node_props}")

    if edges:
        print("\n🔗 Sample edges:")
        for i, edge in enumerate(edges[:3]):
            print(f"  Edge {i + 1}: {edge}")

    # Try different search variations
    print("\n🔍 Testing different search queries...")

    # Get available graphs and collections
    if hasattr(vector_engine, "driver"):
        graphs = vector_engine.driver.list_graphs()
        print(f"Available graphs: {graphs}")

        # Try to query directly to see node labels
        try:
            result = vector_engine.query("MATCH (n) RETURN DISTINCT labels(n) as labels LIMIT 10")
            print(f"Node labels found: {result.result_set}")

            result = vector_engine.query("MATCH (n) RETURN n LIMIT 5")
            print(f"Sample nodes raw: {result.result_set}")

        except Exception as e:
            print(f"Direct query error: {e}")

    # Try searching with different queries
    search_queries = [
        ("entity.name + AI", "entity.name", "AI"),
        ("Entity.name + AI", "Entity.name", "AI"),
        ("text + AI", "text", "AI"),
        ("content + AI", "content", "AI"),
        ("name + AI", "name", "AI"),
    ]

    for query_desc, collection_name, query_text in search_queries:
        try:
            results = await vector_engine.search(
                collection_name=collection_name, query_text=query_text
            )
            print(f"  {query_desc}: {len(results)} results")
            if results:
                print(f"    First result: {results[0]}")
        except Exception as e:
            print(f"  {query_desc}: Error - {e}")

    # Clean up
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("\n✅ Debug completed!")


if __name__ == "__main__":
    asyncio.run(debug_falkordb())
