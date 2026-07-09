"""Unit tests for bounded-subgraph seed resolution and truncation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.visualization.subgraph_data import (
    DEFAULT_NEIGHBORHOOD_DEPTH,
    fetch_visualization_graph_data,
    resolve_seed_node_ids,
    resolve_seeds_from_query,
    resolve_seeds_from_recall,
    truncate_subgraph,
)


def _chain_graph(node_count: int = 20):
    """A simple 0-1-2-...-N chain; deterministic and easy to reason about."""
    nodes = [(str(i), {"type": "Entity", "name": f"N{i}"}) for i in range(node_count)]
    edges = [(str(i), str(i + 1), "related_to", {}) for i in range(node_count - 1)]
    return nodes, edges


# --- recall seed extraction ------------------------------------------------


def test_resolve_seeds_from_recall_node_ids_dict():
    seeds = resolve_seeds_from_recall({"node_ids": ["a", "b", "a"]})
    assert seeds == ["a", "b"]


def test_resolve_seeds_from_recall_qa_entries_provenance():
    entries = [
        SimpleNamespace(used_graph_element_ids={"node_ids": ["9", "8"]}),
        SimpleNamespace(used_graph_element_ids={"node_ids": ["8", "7"]}),
    ]
    seeds = resolve_seeds_from_recall(entries)
    assert seeds == ["9", "8", "7"]


def test_resolve_seeds_from_recall_dict_items():
    # Serialized QA entries (dicts) are handled like their object form.
    items = [{"used_graph_element_ids": {"node_ids": ["9", "8"]}}, {"text": "no provenance"}]
    assert resolve_seeds_from_recall(items) == ["9", "8"]


def test_resolve_seeds_from_recall_ignores_shapes_without_node_ids():
    # A graph result item carries text/metadata but no graph node ids.
    item = SimpleNamespace(text="an answer", metadata={"chunk_id": "c1"})
    assert resolve_seeds_from_recall([item]) == []
    assert resolve_seeds_from_recall(None) == []


def test_resolve_seeds_from_recall_malformed_node_ids_returns_empty():
    # Non-list node_ids must degrade to [] rather than crashing or char-splitting.
    assert resolve_seeds_from_recall({"node_ids": 5}) == []
    assert resolve_seeds_from_recall({"node_ids": "a1b2"}) == []


# --- query seed extraction (ordering fix) ----------------------------------


@pytest.mark.asyncio
async def test_resolve_seeds_from_query_orders_by_distance_and_is_deterministic():
    # Same node "a" appears in two collections with different scores; "a" has
    # the best (lowest) score so it must rank first, and the top-k must be the
    # nearest hits in distance order — not an arbitrary set slice.
    fake = MagicMock()
    fake.embed_and_retrieve_distances = AsyncMock()
    fake.has_results = MagicMock(return_value=True)
    fake.node_distances = {
        "Entity_name": [
            SimpleNamespace(id="c", score=0.9),
            SimpleNamespace(id="a", score=0.1),
        ],
        "DocumentChunk_text": [
            SimpleNamespace(id="b", score=0.5),
            SimpleNamespace(id="a", score=0.3),
        ],
    }

    with patch(
        "cognee.modules.visualization.subgraph_data.NodeEdgeVectorSearch",
        return_value=fake,
    ):
        top2 = await resolve_seeds_from_query("q", seed_top_k=2)
        top3 = await resolve_seeds_from_query("q", seed_top_k=3)

    assert top2 == ["a", "b"]
    assert top3 == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_resolve_seeds_from_query_empty_when_no_hits():
    fake = MagicMock()
    fake.embed_and_retrieve_distances = AsyncMock()
    fake.has_results = MagicMock(return_value=False)
    with patch(
        "cognee.modules.visualization.subgraph_data.NodeEdgeVectorSearch",
        return_value=fake,
    ):
        assert await resolve_seeds_from_query("q") == []


# --- seed priority ---------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_seed_priority_explicit_over_recall_query_degree():
    engine = MagicMock()
    engine.get_graph_data = AsyncMock(return_value=_chain_graph(5))
    with patch(
        "cognee.modules.visualization.subgraph_data.resolve_seeds_from_query",
        AsyncMock(return_value=["q1"]),
    ):
        seeds, source = await resolve_seed_node_ids(
            engine,
            seed_node_ids=["explicit"],
            recall_result={"node_ids": ["r1"]},
            query="ignored",
        )
    assert (seeds, source) == (["explicit"], "explicit")


@pytest.mark.asyncio
async def test_resolve_seed_priority_falls_through_to_degree():
    # Star graph: "hub" has degree 4, every spoke has degree 1.
    nodes = [("hub", {})] + [(f"s{i}", {}) for i in range(4)]
    edges = [("hub", f"s{i}", "rel", {}) for i in range(4)]
    engine = MagicMock()
    engine.get_graph_data = AsyncMock(return_value=(nodes, edges))
    seeds, source = await resolve_seed_node_ids(engine, seed_top_k=1)
    assert source == "degree"
    assert seeds == ["hub"]


@pytest.mark.asyncio
async def test_resolve_seed_none_on_empty_graph():
    engine = MagicMock()
    engine.get_graph_data = AsyncMock(return_value=([], []))
    seeds, source = await resolve_seed_node_ids(engine)
    assert (seeds, source) == ([], "none")


# --- truncation ------------------------------------------------------------


def test_truncate_subgraph_caps_keeps_seeds_and_drops_dangling_edges():
    nodes, edges = _chain_graph(6)
    (kept_nodes, kept_edges), truncated = truncate_subgraph(
        nodes, edges, seed_ids=["0"], max_nodes=3
    )
    kept_ids = {str(n) for n, _ in kept_nodes}
    assert truncated is True
    assert len(kept_nodes) == 3
    assert "0" in kept_ids  # seed always kept
    assert kept_ids == {"0", "1", "2"}  # nearest hop neighbors of the seed
    assert all(str(e[0]) in kept_ids and str(e[1]) in kept_ids for e in kept_edges)


def test_truncate_subgraph_noop_under_cap():
    nodes, edges = _chain_graph(4)
    (kept_nodes, kept_edges), truncated = truncate_subgraph(
        nodes, edges, seed_ids=["0"], max_nodes=500
    )
    assert truncated is False
    assert (kept_nodes, kept_edges) == (nodes, edges)


# --- fetch orchestration ---------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_full_graph_skips_neighborhood():
    full_graph = _chain_graph(5)
    engine = MagicMock()
    engine.get_graph_data = AsyncMock(return_value=full_graph)
    engine.get_neighborhood = AsyncMock()

    graph_data = await fetch_visualization_graph_data(engine, full=True)

    assert graph_data == full_graph
    engine.get_graph_data.assert_awaited_once()
    engine.get_neighborhood.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_subgraph_expands_explicit_seeds():
    full_graph = _chain_graph(20)
    subgraph = (full_graph[0][8:12], full_graph[1][8:11])
    engine = MagicMock()
    engine.get_neighborhood = AsyncMock(return_value=subgraph)

    graph_data = await fetch_visualization_graph_data(
        engine,
        seed_node_ids=["10"],
        neighborhood_depth=DEFAULT_NEIGHBORHOOD_DEPTH,
    )

    engine.get_neighborhood.assert_awaited_once_with(
        node_ids=["10"], depth=DEFAULT_NEIGHBORHOOD_DEPTH
    )
    assert graph_data == subgraph


@pytest.mark.asyncio
async def test_fetch_subgraph_uses_query_seeds():
    subgraph = _chain_graph(4)
    engine = MagicMock()
    engine.get_neighborhood = AsyncMock(return_value=subgraph)
    engine.get_graph_data = AsyncMock()

    with patch(
        "cognee.modules.visualization.subgraph_data.resolve_seeds_from_query",
        AsyncMock(return_value=["3", "4"]),
    ):
        await fetch_visualization_graph_data(engine, query="python")

    engine.get_neighborhood.assert_awaited_once_with(node_ids=["3", "4"], depth=2)
    engine.get_graph_data.assert_not_awaited()  # query path never loads full graph


@pytest.mark.asyncio
async def test_fetch_truncates_oversized_neighborhood():
    # get_neighborhood returns more than max_nodes; fetch must cap while keeping
    # the seed and leaving no dangling edges.
    nodes, edges = _chain_graph(20)
    engine = MagicMock()
    engine.get_neighborhood = AsyncMock(return_value=(nodes, edges))

    kept_nodes, kept_edges = await fetch_visualization_graph_data(
        engine, seed_node_ids=["0"], neighborhood_depth=5, max_nodes=4
    )
    kept_ids = {str(n) for n, _ in kept_nodes}
    assert len(kept_nodes) == 4
    assert "0" in kept_ids
    assert all(str(e[0]) in kept_ids and str(e[1]) in kept_ids for e in kept_edges)


@pytest.mark.asyncio
async def test_fetch_no_seeds_renders_empty():
    engine = MagicMock()
    engine.get_graph_data = AsyncMock(return_value=([], []))
    engine.get_neighborhood = AsyncMock()

    graph_data = await fetch_visualization_graph_data(engine)

    assert graph_data == ([], [])
    engine.get_neighborhood.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"neighborhood_depth": 0}, {"seed_top_k": 0}, {"max_nodes": 0}],
)
async def test_fetch_validates_bounds(kwargs):
    engine = MagicMock()
    with pytest.raises(ValueError):
        await fetch_visualization_graph_data(engine, **kwargs)
