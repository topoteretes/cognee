"""`get_triplets_batch` pages the whole graph, so its result order has to be stable.

`get_triplet_datapoints` walks every triplet with a `while True` loop that raises
`offset` by the batch size and stops on the first empty batch. That is only correct
if two calls with different offsets read from the same ordering. PostgresDemoAdapter and
TursoVectorAdapter order explicitly (`ORDER BY e.source_id, e.target_id,
e.relationship_name`) before applying `OFFSET`/`LIMIT`; the two Cypher adapters
paginated an unordered match, where row order is not defined.
"""

from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter


def assert_orders_before_paginating(query: str) -> None:
    """The ordering has to be applied to the stream that SKIP/LIMIT then slices."""
    assert "ORDER BY" in query, f"no ordering in paginated query:\n{query}"
    assert query.index("ORDER BY") < query.index("SKIP"), (
        f"SKIP is applied before the ordering:\n{query}"
    )


@pytest.mark.asyncio
async def test_ladybug_triplets_batch_orders_before_paginating():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = AsyncMock(return_value=[])

    await adapter.get_triplets_batch(offset=10, limit=5)

    query, params = adapter.query.await_args.args
    assert params == {"offset": 10, "limit": 5}
    assert_orders_before_paginating(query)
    assert "start_node.id, end_node.id, relationship.relationship_name" in query


@pytest.mark.asyncio
async def test_neo4j_triplets_batch_orders_before_paginating():
    pytest.importorskip("neo4j")

    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    adapter = Neo4jAdapter(
        "bolt://unused",
        graph_database_allow_anonymous=True,
        driver=object(),
    )
    adapter.query = AsyncMock(return_value=[])

    await adapter.get_triplets_batch(offset=10, limit=5)

    query, params = adapter.query.await_args.args
    assert params == {"offset": 10, "limit": 5}
    assert_orders_before_paginating(query)
    # relationship_name is the relationship type in this adapter, not a property.
    assert "start_node.id, end_node.id, type(relationship)" in query
