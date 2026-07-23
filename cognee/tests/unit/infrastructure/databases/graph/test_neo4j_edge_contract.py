from unittest.mock import AsyncMock

import pytest

pytest.importorskip("neo4j")

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


def _make_adapter() -> Neo4jAdapter:
    return Neo4jAdapter(
        "bolt://unused",
        graph_database_allow_anonymous=True,
        driver=object(),
    )


@pytest.mark.asyncio
async def test_has_edges_returns_existing_edge_tuples():
    """has_edges must return the subset of input edges that exist, as
    (from_node, to_node, relationship_name) tuples — the graph_db_interface
    contract shared with the ladybug/postgres/neptune adapters — not a list
    of booleans."""
    adapter = _make_adapter()
    adapter.query = AsyncMock(
        return_value=[
            {
                "from_node": "node-a",
                "to_node": "node-b",
                "relationship_name": "knows",
                "edge_exists": True,
            },
            {
                "from_node": "node-a",
                "to_node": "node-c",
                "relationship_name": "likes",
                "edge_exists": False,
            },
        ]
    )

    result = await adapter.has_edges([("node-a", "node-b", "knows"), ("node-a", "node-c", "likes")])

    assert result == [("node-a", "node-b", "knows")]


@pytest.mark.asyncio
async def test_has_edges_matches_on_id_property_not_internal_id():
    """Nodes are stored with an ``id`` property (a UUID string); Neo4j's
    internal ``id()`` is an integer and never equals it, which made the
    previous query match nothing and existing-edge deduplication silently
    return empty."""
    adapter = _make_adapter()
    adapter.query = AsyncMock(return_value=[])

    await adapter.has_edges([("node-a", "node-b", "knows")])

    query = adapter.query.await_args.args[0]
    assert "a.id = edge.from_node" in query
    assert "b.id = edge.to_node" in query
    assert "id(a)" not in query
    assert "id(b)" not in query


@pytest.mark.asyncio
async def test_get_connections_preserves_edge_properties():
    """get_connections must merge the relationship's properties into the edge
    dict (as the postgres/ladybug/neptune adapters do), not reduce it to just
    the relationship name — consumers such as legacy_delete read keys like
    ``edge_text`` from it."""
    adapter = _make_adapter()
    relation_tuple = ({"id": "node-a"}, "knows", {"id": "node-b"})
    adapter.query = AsyncMock(
        side_effect=[
            [
                {
                    "relation": relation_tuple,
                    "relation_properties": {"edge_text": "a knows b", "weight": 0.7},
                }
            ],
            [],
        ]
    )

    connections = await adapter.get_connections("node-b")

    assert connections == [
        (
            {"id": "node-a"},
            {"relationship_name": "knows", "edge_text": "a knows b", "weight": 0.7},
            {"id": "node-b"},
        )
    ]


@pytest.mark.asyncio
async def test_get_connections_handles_missing_edge_properties():
    """Edges without extra properties still yield a relationship_name-only dict."""
    adapter = _make_adapter()
    relation_tuple = ({"id": "node-a"}, "knows", {"id": "node-b"})
    adapter.query = AsyncMock(
        side_effect=[
            [],
            [{"relation": relation_tuple, "relation_properties": {}}],
        ]
    )

    connections = await adapter.get_connections("node-a")

    assert connections == [({"id": "node-a"}, {"relationship_name": "knows"}, {"id": "node-b"})]
