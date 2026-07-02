from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.visualization.subgraph_data import (
    DEFAULT_MAX_NODES,
    DEFAULT_NEIGHBORHOOD_DEPTH,
    fetch_visualization_graph_data,
    resolve_seed_node_ids,
    resolve_seeds_from_recall,
    truncate_subgraph,
)


def _chain_graph(node_count: int = 20):
    nodes = [(str(i), {"type": "Entity", "name": f"N{i}"}) for i in range(node_count)]
    edges = [(str(i), str(i + 1), "related_to", {}) for i in range(node_count - 1)]
    return nodes, edges


def test_resolve_seeds_from_recall_dict_node_ids():
    seeds = resolve_seeds_from_recall({"node_ids": ["a", "b", "a"]})
    assert seeds == ["a", "b"]


def test_resolve_seeds_from_recall_edge_list():
    edge = SimpleNamespace(
        node1=SimpleNamespace(id="1"),
        node2=SimpleNamespace(id="2"),
        attributes={"edge_object_id": "e1"},
    )
    seeds = resolve_seeds_from_recall([edge])
    assert seeds == ["1", "2"]


def test_resolve_seeds_from_recall_qa_entry():
    entry = SimpleNamespace(used_graph_element_ids={"node_ids": ["9", "8"]})
    seeds = resolve_seeds_from_recall([entry])
    assert seeds == ["9", "8"]


def test_truncate_subgraph_honors_max_nodes_and_keeps_seeds():
    nodes, edges = _chain_graph(6)
    graph_data, truncated = truncate_subgraph(nodes, edges, seed_ids=["0"], max_nodes=3)
    kept_nodes, kept_edges = graph_data
    assert truncated is True
    assert len(kept_nodes) == 3
    assert any(str(node_id) == "0" for node_id, _ in kept_nodes)
    assert all(str(edge[0]) in {str(n[0]) for n in kept_nodes} for edge in kept_edges)


@pytest.mark.asyncio
async def test_resolve_seed_priority_explicit_over_query():
    engine = MagicMock()
    with patch(
        "cognee.modules.visualization.subgraph_data.resolve_seeds_from_query",
        AsyncMock(return_value=["q1"]),
    ):
        seeds, source = await resolve_seed_node_ids(
            engine,
            seed_node_ids=["explicit"],
            query="ignored",
        )
    assert seeds == ["explicit"]
    assert source == "explicit"


@pytest.mark.asyncio
async def test_fetch_full_graph_when_full_true():
    engine = MagicMock()
    full_graph = _chain_graph(5)
    engine.get_graph_data = AsyncMock(return_value=full_graph)
    engine.get_neighborhood = AsyncMock()

    graph_data, meta = await fetch_visualization_graph_data(engine, full=True)
    assert graph_data == full_graph
    assert meta.scope == "all"
    engine.get_graph_data.assert_awaited_once()
    engine.get_neighborhood.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_subgraph_calls_neighborhood_without_edge_types():
    engine = MagicMock()
    full_graph = _chain_graph(20)
    subgraph = (full_graph[0][8:12], full_graph[1][8:11])
    engine.get_graph_data = AsyncMock(return_value=full_graph)
    engine.get_neighborhood = AsyncMock(return_value=subgraph)
    engine.query = AsyncMock(return_value=[])

    with patch(
        "cognee.modules.visualization.session_events.get_latest_session_seed_node_ids",
        AsyncMock(return_value=[]),
    ):
        graph_data, meta = await fetch_visualization_graph_data(
            engine,
            seed_node_ids=["10"],
            neighborhood_depth=DEFAULT_NEIGHBORHOOD_DEPTH,
            max_nodes=DEFAULT_MAX_NODES,
        )

    engine.get_neighborhood.assert_awaited_once()
    call_kwargs = engine.get_neighborhood.await_args.kwargs
    assert call_kwargs["node_ids"] == ["10"]
    assert call_kwargs["depth"] == DEFAULT_NEIGHBORHOOD_DEPTH
    assert "edge_types" not in call_kwargs
    assert meta.scope == "subgraph"
    assert meta.seed_source == "explicit"
    assert graph_data == subgraph


@pytest.mark.asyncio
async def test_fetch_subgraph_uses_query_seeds():
    engine = MagicMock()
    subgraph = _chain_graph(4)
    engine.get_neighborhood = AsyncMock(return_value=subgraph)
    engine.query = AsyncMock(return_value=[])

    with (
        patch(
            "cognee.modules.visualization.subgraph_data.resolve_seeds_from_query",
            AsyncMock(return_value=["3", "4"]),
        ),
        patch(
            "cognee.modules.visualization.session_events.get_latest_session_seed_node_ids",
            AsyncMock(return_value=[]),
        ),
    ):
        _, meta = await fetch_visualization_graph_data(engine, query="python")

    assert meta.seed_source == "query"
    engine.get_neighborhood.assert_awaited_once_with(node_ids=["3", "4"], depth=2)


@pytest.mark.asyncio
async def test_fetch_subgraph_degree_fallback():
    engine = MagicMock()
    subgraph = _chain_graph(3)
    engine.get_neighborhood = AsyncMock(return_value=subgraph)
    engine.query = AsyncMock(side_effect=NotImplementedError())

    full_graph = _chain_graph(5)
    engine.get_graph_data = AsyncMock(return_value=full_graph)

    with patch(
        "cognee.modules.visualization.session_events.get_latest_session_seed_node_ids",
        AsyncMock(return_value=[]),
    ):
        _, meta = await fetch_visualization_graph_data(engine)

    assert meta.seed_source == "degree"
    engine.get_neighborhood.assert_awaited_once()
