import asyncio
import time
from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter

# This will create the test.db if it doesn't exist


async def main():
    adapter = KuzuAdapter("test.db")
    result = await adapter.query("MATCH (n:Node) RETURN COUNT(n)")
    print(f"Reader: Found {result[0][0]} nodes")
    result = await adapter.query("MATCH (n:Node) RETURN COUNT(n)")
    print(f"Reader: Found {result[0][0]} nodes")
    result = await adapter.query("MATCH (n:Node) RETURN COUNT(n)")
    print(f"Reader: Found {result[0][0]} nodes")
    result = await adapter.query("MATCH (n:Node) RETURN COUNT(n)")
    print(f"Reader: Found {result[0][0]} nodes")
    result = await adapter.query("MATCH (n:Node) RETURN COUNT(n)")
    print(f"Reader: Found {result} nodes")
    result = await adapter.query("MATCH (n:Node) RETURN COUNT(n)")
    print(f"Reader: Found {result[0][0]} nodes")


if __name__ == "__main__":
    asyncio.run(main())
