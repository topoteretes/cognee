import asyncio
from cognee.infrastructure.databases.graph.kuzu.remote_kuzu_adapter import RemoteKuzuAdapter
from cognee.infrastructure.databases.graph.config import get_graph_config


async def main():
    config = get_graph_config()
    adapter = RemoteKuzuAdapter(
        config.graph_database_url, config.graph_database_username, config.graph_database_password
    )
    try:
        print("Node Count:")
        result = await adapter.query("MATCH (n) RETURN COUNT(n) as count")
        print(result)

        print("\nEdge Count:")
        result = await adapter.query("MATCH ()-[r]->() RETURN COUNT(r) as count")
        print(result)

        print("\nSample Nodes with Properties:")
        result = await adapter.query("MATCH (n) RETURN n LIMIT 5")
        print(result)

        print("\nSample Relationships with Properties:")
        result = await adapter.query("MATCH (n1)-[r]->(n2) RETURN n1, r, n2 LIMIT 5")
        print(result)

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
