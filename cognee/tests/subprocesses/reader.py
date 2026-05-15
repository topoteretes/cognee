import asyncio

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from common import get_kuzu_db_path


async def main():
    adapter = LadybugAdapter(get_kuzu_db_path())

    for _ in range(5):
        result = await adapter.query("MATCH (n:Node) RETURN COUNT(n)")
        print(f"Reader: Found {result[0][0]} nodes")

    result = await adapter.query("MATCH (n:Node) RETURN COUNT(n)")
    print(f"Reader: Found {result} nodes")


if __name__ == "__main__":
    asyncio.run(main())
