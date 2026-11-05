"""Ladybug connected-component metrics must be global and complete (issue #3491).

``get_graph_metrics`` used to derive ``num_connected_components`` /
``sizes_of_connected_components`` from ``(n)-[:EDGE*1..3]-(m)`` expansions
grouped by start node, which split components whose members sit more than 3
hops apart, counted roughly one "component" per node's local neighborhood, and
never counted isolated (degree-0) nodes (the smallest expansion is 1 hop). The
fix computes components with union-find over the full node-id and edge lists.
"""

from unittest.mock import AsyncMock

import pytest

NODE_ID_QUERY_PREFIX = "MATCH (n:Node) RETURN n.id"
EDGE_QUERY_PREFIX = "MATCH (a:Node)-[:EDGE]->(b:Node) RETURN a.id, b.id"


def _adapter_with_graph(nodes, edges):
    """A LadybugAdapter whose ``query`` answers only the component queries.

    Any other query (e.g. the old ``*1..3`` expansion) returns no rows, so the
    previous implementation reports 0 components / [] and fails these tests.
    """

    async def dispatch(query, params=None):
        if query.startswith(NODE_ID_QUERY_PREFIX):
            return [(node_id,) for node_id in nodes]
        if query.startswith(EDGE_QUERY_PREFIX):
            return list(edges)
        return []

    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = AsyncMock(side_effect=dispatch)
    return adapter


@pytest.mark.asyncio
async def test_chain_longer_than_three_hops_is_one_component():
    """A → B → … → F spans 5 hops; the old 3-hop cap split it into pieces."""
    nodes = ["a", "b", "c", "d", "e", "f", "isolated"]
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")]

    adapter = _adapter_with_graph(nodes, edges)

    assert await adapter._get_num_connected_components() == 2
    assert await adapter._get_size_of_connected_components() == [6, 1]


@pytest.mark.asyncio
async def test_isolated_nodes_count_as_components():
    """Degree-0 nodes are components of size 1; the old *1..3 expansion never saw them."""
    adapter = _adapter_with_graph(["x", "y", "z"], [])

    assert await adapter._get_num_connected_components() == 3
    assert await adapter._get_size_of_connected_components() == [1, 1, 1]


@pytest.mark.asyncio
async def test_sizes_are_sorted_descending_and_selfloops_are_neutral():
    nodes = ["a", "b", "c", "d", "e"]
    edges = [("a", "b"), ("c", "d"), ("c", "c")]

    adapter = _adapter_with_graph(nodes, edges)

    assert await adapter._get_size_of_connected_components() == [2, 2, 1]
    assert await adapter._get_num_connected_components() == 3


@pytest.mark.asyncio
async def test_get_graph_metrics_reports_component_fields_consistently():
    nodes = ["a", "b", "c", "isolated"]
    edges = [("a", "b"), ("b", "c")]

    adapter = _adapter_with_graph(nodes, edges)
    adapter.get_graph_counts = AsyncMock(return_value=(len(nodes), len(edges)))

    metrics = await adapter.get_graph_metrics()

    assert metrics["num_connected_components"] == 2
    assert metrics["sizes_of_connected_components"] == [3, 1]
    assert metrics["num_connected_components"] == len(metrics["sizes_of_connected_components"])


@pytest.mark.asyncio
async def test_components_against_real_ladybug_database(tmp_path):
    """End-to-end on the embedded engine: 4-node chain plus an isolated node."""
    pytest.importorskip("ladybug")

    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    adapter = LadybugAdapter(db_path=str(tmp_path / "components.lbug"))
    try:
        for node_id in ["node-0", "node-1", "node-2", "node-3", "node-lone"]:
            await adapter.query(
                "CREATE (:Node {id: $id, name: $id, type: 'Entity', properties: '{}'})",
                {"id": node_id},
            )
        for source, target in [("node-0", "node-1"), ("node-1", "node-2"), ("node-2", "node-3")]:
            await adapter.query(
                "MATCH (a:Node {id: $s}), (b:Node {id: $t}) "
                "CREATE (a)-[:EDGE {relationship_name: 'knows', properties: '{}'}]->(b)",
                {"s": source, "t": target},
            )

        metrics = await adapter.get_graph_metrics()

        assert metrics["num_connected_components"] == 2
        assert metrics["sizes_of_connected_components"] == [4, 1]
    finally:
        await adapter.close()
