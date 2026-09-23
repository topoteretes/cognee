from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


@pytest.mark.asyncio
async def test_has_node_awaits_query_and_returns_true():
    """Regression test: has_node must await the async query().

    The bug called ``self.query(...)`` without ``await``, so ``results`` was a
    coroutine and ``len(results)`` raised ``TypeError`` — has_node could never
    return a value (and leaked an un-awaited coroutine).
    """
    adapter = object.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[{"node_exists": True}])

    assert await adapter.has_node("node-1") is True
    adapter.query.assert_awaited_once()


@pytest.mark.asyncio
async def test_has_node_returns_false_when_no_results():
    adapter = object.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[])

    assert await adapter.has_node("missing") is False
