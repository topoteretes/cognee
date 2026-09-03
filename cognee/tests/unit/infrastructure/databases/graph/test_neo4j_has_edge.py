from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


@pytest.mark.asyncio
async def test_has_edge_returns_bool_true_when_present():
    adapter = object.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[{"edge_exists": True}])

    assert await adapter.has_edge("node-a", "node-b", "relates_to") is True


@pytest.mark.asyncio
async def test_has_edge_returns_bool_false_when_absent():
    """Regression: the query is an aggregation that always returns exactly one row,
    {"edge_exists": False} when the edge is absent. Returning the raw result list
    made every call truthy, so `if not has_edge(...)` never saw a missing edge —
    cross_connect_entities' _drop_existing_edges dropped every candidate and the
    memify pipeline wrote zero inferred edges on Neo4j."""
    adapter = object.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[{"edge_exists": False}])

    result = await adapter.has_edge("node-a", "node-b", "relates_to")

    assert isinstance(result, bool)
    assert result is False
    assert not result  # the caller-facing check


@pytest.mark.asyncio
async def test_has_edge_returns_false_on_empty_result():
    adapter = object.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[])

    assert await adapter.has_edge("node-a", "node-b", "relates_to") is False


@pytest.mark.asyncio
async def test_has_edge_query_binds_the_relationship_variable():
    """Regression for the Cypher syntax error: COUNT(relationship) referenced a
    variable the pattern never introduced (`-[:`label`]->` binds nothing), so
    Neo4j 5 raised `Variable relationship not defined` on every call. The pattern
    must bind the relationship for the count expression to be valid Cypher."""
    adapter = object.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[{"edge_exists": False}])

    await adapter.has_edge("node-a", "node-b", "relates_to")

    query = adapter.query.await_args.args[0]
    assert "-[relationship: `relates_to`]->" in query
    assert "COUNT(relationship)" in query
