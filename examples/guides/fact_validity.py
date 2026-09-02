import asyncio

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.models import Entity

# close_node and is_valid are not re-exported from cognee.tasks.storage,
# so import them from their module.
from cognee.tasks.storage.close_node import close_node, is_valid


async def main():
    # Step 1: start clean and remember the original fact.
    await cognee.forget(everything=True)
    await cognee.remember("Alice works at Acme.", dataset_name="employment_facts")

    # Step 2: entity node ids are deterministic — derive Acme's id from its name.
    acme_id = Entity.id_for("Acme")

    # Step 3: Alice changes jobs — close the old fact instead of deleting it.
    closed = await close_node(acme_id)
    print("closed:", closed)  # True — the node existed and valid_to was stamped

    # Step 4: remember the replacement fact.
    await cognee.remember("Alice works at Globex.", dataset_name="employment_facts")

    # Step 5: the closed node is still in the graph, just stale.
    graph = await get_graph_engine()
    node = await graph.get_node(str(acme_id))
    print("is_valid:", is_valid(node))  # False — the fact was superseded


if __name__ == "__main__":
    asyncio.run(main())
