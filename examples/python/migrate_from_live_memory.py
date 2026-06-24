"""Migrate memory from running Mem0 or Graphiti instances into Cognee.

Requires optional extras and environment variables::

    pip install cognee[mem0,graphiti]
    export MEM0_API_KEY=...
    export GRAPH_DATABASE_URL=bolt://localhost:7687
    export GRAPH_DATABASE_PASSWORD=...

Run::

    python examples/python/migrate_from_live_memory.py
"""

import asyncio
import os
import sys


async def migrate_from_mem0(dataset_name: str = "migrated_mem0") -> None:
    api_key = os.environ.get("MEM0_API_KEY")
    user_id = os.environ.get("MEM0_USER_ID")
    if not api_key or not user_id:
        print("Skipping Mem0: set MEM0_API_KEY and MEM0_USER_ID.", file=sys.stderr)
        return

    from mem0 import MemoryClient

    import cognee
    from cognee.migration import Mem0LiveSource

    client = MemoryClient(api_key=api_key)
    source = Mem0LiveSource(client=client, filters={"user_id": user_id})
    result = await cognee.remember(source, dataset_name=dataset_name)
    print("Mem0 import:", result.items[-1] if result.items else result)


async def migrate_from_graphiti(dataset_name: str = "migrated_graphiti") -> None:
    url = os.environ.get("GRAPH_DATABASE_URL")
    password = os.environ.get("GRAPH_DATABASE_PASSWORD", "")
    if not url:
        print("Skipping Graphiti: set GRAPH_DATABASE_URL.", file=sys.stderr)
        return

    from graphiti_core import Graphiti

    import cognee
    from cognee.migration import GraphitiLiveSource

    graphiti = Graphiti(url, os.environ.get("GRAPH_DATABASE_USERNAME", "neo4j"), password)
    try:
        source = GraphitiLiveSource(graphiti=graphiti, mode="hybrid")
        result = await cognee.remember(source, dataset_name=dataset_name)
        print("Graphiti import:", result.items[-1] if result.items else result)
    finally:
        await graphiti.close()


async def main() -> None:
    await migrate_from_mem0()
    await migrate_from_graphiti()


if __name__ == "__main__":
    asyncio.run(main())
