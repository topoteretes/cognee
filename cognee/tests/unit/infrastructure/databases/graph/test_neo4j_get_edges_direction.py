"""Unit tests for Neo4jAdapter.get_edges true-direction tuples (issue #4967).

get_edges used to return (queried_node, neighbor, ...) for every row, which
inverted incoming edges. Now the tuple must follow startNode(r)/endNode(r)
regardless of which side of the pattern the queried node landed on.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    import neo4j  # noqa: F401
except ModuleNotFoundError:
    sys.modules["neo4j"] = MagicMock()
    sys.modules["neo4j.exceptions"] = MagicMock()

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


def _adapter_returning(rows):
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=rows)
    return adapter


@pytest.mark.asyncio
async def test_outgoing_edge_keeps_direction():
    # stored: (node-a) -[:reports_to]-> (node-b); queried node-a
    rows = [
        {
            "r": (123, "reports_to"),
            "source_id": "node-a",
            "target_id": "node-b",
        }
    ]
    edges = await _adapter_returning(rows).get_edges("node-a")
    assert edges == [("node-a", "node-b", {"relationship_name": "reports_to"})]


@pytest.mark.asyncio
async def test_incoming_edge_is_not_inverted():
    # stored: (node-b) -[:reports_to]-> (node-a); queried node-a.
    # Cypher still binds n=node-a (undirected match), but startNode is node-b.
    rows = [
        {
            "r": (123, "reports_to"),
            "source_id": "node-b",
            "target_id": "node-a",
        }
    ]
    edges = await _adapter_returning(rows).get_edges("node-a")
    assert edges == [("node-b", "node-a", {"relationship_name": "reports_to"})]


@pytest.mark.asyncio
async def test_query_projects_start_and_end_nodes():
    adapter = _adapter_returning([])
    await adapter.get_edges("node-a")
    cypher = adapter.query.call_args[0][0]
    assert "startNode(r).id AS source_id" in cypher
    assert "endNode(r).id AS target_id" in cypher
    # dead projections removed: only what the tuple needs crosses the wire
    assert "RETURN n, r, m" not in cypher
