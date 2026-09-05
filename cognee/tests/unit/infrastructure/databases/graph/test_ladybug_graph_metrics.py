"""Tests for LadybugAdapter graph-metrics connected-component computation.

Regression coverage for the bug where ``_get_num_connected_components`` /
``_get_size_of_connected_components`` walked a bounded ``[:EDGE*1..3]`` neighborhood
per node and counted those per-node neighborhoods as components — so they ignored
connectivity beyond 3 hops, never collapsed a component to one entry, and dropped
isolated nodes.
"""

from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import (
    LadybugAdapter,
    _weakly_connected_component_sizes,
)


# --------------------------------------------------------------------------- #
# _weakly_connected_component_sizes (pure)
# --------------------------------------------------------------------------- #


def test_component_sizes_pairs_chain_and_isolated():
    # Three 2-node pairs, a 5-node chain (spans 4 hops), and one isolated node.
    node_ids = ["A", "B", "C", "D", "E", "F", "G", "P1", "P2", "P3", "P4", "P5"]
    edges = [
        ("A", "B"),
        ("C", "D"),
        ("E", "F"),
        ("P1", "P2"),
        ("P2", "P3"),
        ("P3", "P4"),
        ("P4", "P5"),
    ]
    assert _weakly_connected_component_sizes(node_ids, edges) == [5, 2, 2, 2, 1]


def test_component_sizes_chain_beyond_three_hops_is_one_component():
    # A 6-node chain is a single component even though endpoints are 5 hops apart.
    node_ids = ["n0", "n1", "n2", "n3", "n4", "n5"]
    edges = [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5")]
    assert _weakly_connected_component_sizes(node_ids, edges) == [6]


def test_component_sizes_all_isolated():
    assert _weakly_connected_component_sizes(["a", "b", "c"], []) == [1, 1, 1]


def test_component_sizes_edges_are_undirected():
    assert _weakly_connected_component_sizes(["a", "b"], [("b", "a")]) == [2]


def test_component_sizes_empty():
    assert _weakly_connected_component_sizes([], []) == []


def test_component_sizes_ignores_unknown_endpoints():
    assert _weakly_connected_component_sizes(["a", "b"], [("a", "missing")]) == [1, 1]


def test_component_sizes_self_loop_stays_singleton():
    assert _weakly_connected_component_sizes(["a", "b"], [("a", "a")]) == [1, 1]


# --------------------------------------------------------------------------- #
# adapter methods (DB I/O mocked)
# --------------------------------------------------------------------------- #


def _adapter_with_graph(node_ids, edge_pairs):
    """A LadybugAdapter whose query() returns a fixed node/edge graph."""
    adapter = LadybugAdapter.__new__(LadybugAdapter)

    async def fake_query(query, params=None):
        if "RETURN n.id" in query:
            return [[node_id] for node_id in node_ids]
        if "a.id, b.id" in query:
            return [[source, target] for source, target in edge_pairs]
        if "COUNT(n)" in query:
            return [[len(node_ids)]]
        if "COUNT(r)" in query:
            return [[len(edge_pairs)]]
        return []

    adapter.query = AsyncMock(side_effect=fake_query)
    return adapter


_GRAPH = (
    ["A", "B", "C", "D", "E", "F", "G", "P1", "P2", "P3", "P4", "P5"],
    [
        ("A", "B"),
        ("C", "D"),
        ("E", "F"),
        ("P1", "P2"),
        ("P2", "P3"),
        ("P3", "P4"),
        ("P4", "P5"),
    ],
)


@pytest.mark.asyncio
async def test_get_size_of_connected_components_uses_full_graph():
    adapter = _adapter_with_graph(*_GRAPH)
    assert await adapter._get_size_of_connected_components() == [5, 2, 2, 2, 1]


@pytest.mark.asyncio
async def test_get_num_connected_components_matches_sizes():
    adapter = _adapter_with_graph(*_GRAPH)
    assert await adapter._get_num_connected_components() == 5


@pytest.mark.asyncio
async def test_get_graph_metrics_reports_correct_components():
    adapter = _adapter_with_graph(*_GRAPH)
    metrics = await adapter.get_graph_metrics()

    assert metrics["num_nodes"] == 12
    assert metrics["num_edges"] == 7
    assert metrics["num_connected_components"] == 5
    assert metrics["sizes_of_connected_components"] == [5, 2, 2, 2, 1]
