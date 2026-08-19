from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


@pytest.mark.asyncio
async def test_has_edge_returns_bool_true():
    """Regression test: has_edge must return a bool, not the raw query result.

    The bug returned the query result list directly (always truthy when
    non-empty) instead of the ``edge_exists`` value, and its Cypher referenced
    an unbound ``relationship`` identifier (the relationship had no alias).
    """
    adapter = object.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[{"edge_exists": True}])

    result = await adapter.has_edge(uuid4(), uuid4(), "KNOWS")

    assert result is True
    adapter.query.assert_awaited_once()


@pytest.mark.asyncio
async def test_has_edge_returns_false_when_no_results():
    adapter = object.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[])

    assert await adapter.has_edge(uuid4(), uuid4(), "KNOWS") is False
