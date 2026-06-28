"""Unit tests for centrality retriever — no LLM, no graph DB, no API key needed."""

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge
from cognee.modules.retrieval.centrality_retriever import (
    CentralityRetriever,
    _compute_degree_centrality,
    _compute_pagerank,
    EXCLUDED_NODE_TYPES,
)


# ---------------------------------------------------------------------------
# Helper — build a tiny directed graph we know the answers for.
#
# `add_edge` adds each edge to BOTH node1 and node2 skeleton_edges, so the
# count is total incident edges (in + out), not just outgoing.
#
#   A ──→ B
#   │     │
#   ↓     ↓
#   C ←───┘     (B→C)
#   │
#   ↓
#   D           (C→D)
#
# skeleton_edges (add_edge adds to both ends):
#   A: A→B  +  A→C       = 2
#   B: A→B  +  B→C       = 2
#   C: A→B  +  B→C + C→D = 3   ← hub
#   D: C→D                = 1
# ---------------------------------------------------------------------------


def _build_test_graph() -> CogneeGraph:
    g = CogneeGraph(directed=True)

    a = Node("A", {"type": "Entity", "name": "node_a", "description": "alpha"})
    b = Node("B", {"type": "Entity", "name": "node_b", "description": "bravo"})
    c = Node("C", {"type": "Entity", "name": "node_c", "description": "charlie"})
    d = Node("D", {"type": "Entity", "name": "node_d", "description": "delta"})

    for n in (a, b, c, d):
        g.add_node(n)

    for e in (
        Edge(a, b, directed=True),
        Edge(a, c, directed=True),
        Edge(b, c, directed=True),
        Edge(c, d, directed=True),
    ):
        g.add_edge(e)  # <-- already calls add_skeleton_edge on both ends

    return g


def _build_graph_with_entity_type() -> CogneeGraph:
    """Like _build_test_graph but includes an EntityType label node."""
    g = CogneeGraph(directed=True)

    a = Node("A", {"type": "Entity", "name": "node_a"})
    b = Node("B", {"type": "Entity", "name": "node_b"})
    svc = Node("Service", {"type": "EntityType", "name": "Service"})

    for n in (a, b, svc):
        g.add_node(n)

    for e in (
        Edge(a, b, directed=True),
        Edge(svc, a, directed=True),
        Edge(svc, b, directed=True),
    ):
        g.add_edge(e)

    return g


# ===== _compute_degree_centrality ==========================================


class TestComputeDegreeCentrality:
    def test_degree_returns_expected_ranking(self):
        g = _build_test_graph()
        scores = _compute_degree_centrality(g, list(g.nodes.keys()))
        assert scores["A"] == 2.0
        assert scores["B"] == 2.0
        assert scores["C"] == 3.0  # hub
        assert scores["D"] == 1.0

    def test_degree_empty_node_list(self):
        g = _build_test_graph()
        scores = _compute_degree_centrality(g, [])
        assert scores == {}

    def test_degree_unknown_node_returns_zero(self):
        g = _build_test_graph()
        scores = _compute_degree_centrality(g, ["GHOST"])
        assert scores["GHOST"] == 0.0


# ===== _compute_pagerank ===================================================


class TestComputePageRank:
    def test_pagerank_converges_and_sums_to_one(self):
        g = _build_test_graph()
        scores = _compute_pagerank(g, list(g.nodes.keys()))
        assert abs(sum(scores.values()) - 1.0) < 1e-4
        assert all(v >= 0 for v in scores.values())

    def test_pagerank_empty_node_list(self):
        g = _build_test_graph()
        scores = _compute_pagerank(g, [])
        assert scores == {}

    def test_pagerank_single_isolated_node(self):
        g = CogneeGraph()
        g.add_node(Node("X", {"type": "Entity"}))
        scores = _compute_pagerank(g, ["X"])
        assert abs(scores["X"] - 1.0) < 1e-4


# ===== _is_entity_node =====================================================


class TestIsEntityNode:
    def test_entity_node_included(self):
        retriever = CentralityRetriever()
        entity = Node("x", {"type": "Entity"})
        assert retriever._is_entity_node(entity)

    def test_entitytype_node_excluded(self):
        retriever = CentralityRetriever()
        label = Node("x", {"type": "EntityType"})
        assert not retriever._is_entity_node(label)

    def test_nodeset_node_excluded(self):
        retriever = CentralityRetriever()
        ns = Node("x", {"type": "NodeSet"})
        assert not retriever._is_entity_node(ns)

    def test_empty_type_excluded(self):
        retriever = CentralityRetriever()
        empty = Node("x", {"type": ""})
        assert not retriever._is_entity_node(empty)

    def test_excluded_constant_matches_implementation(self):
        assert "EntityType" in EXCLUDED_NODE_TYPES
        assert "NodeSet" in EXCLUDED_NODE_TYPES


# ===== get_context_from_objects (async) ====================================


@pytest.mark.asyncio
class TestGetContextFromObjects:
    async def test_formats_ranked_list(self):
        retriever = CentralityRetriever()
        results = [
            {"node_name": "node_a", "score": 2.0, "description": "alpha"},
            {"node_name": "node_b", "score": 1.0, "description": ""},
        ]
        text = await retriever.get_context_from_objects(retrieved_objects=results)
        assert "node_a" in text
        assert "2.0" in text
        assert "alpha" in text
        assert "node_b" in text

    async def test_empty_results_returns_empty_string(self):
        retriever = CentralityRetriever()
        assert await retriever.get_context_from_objects(retrieved_objects=[]) == ""

    async def test_none_results_returns_empty_string(self):
        retriever = CentralityRetriever()
        assert await retriever.get_context_from_objects(retrieved_objects=None) == ""


# ===== get_completion_from_context (async) =================================


@pytest.mark.asyncio
class TestGetCompletionFromContext:
    async def test_returns_list_with_header_and_context(self):
        retriever = CentralityRetriever()
        result = await retriever.get_completion_from_context(
            query="test query", context="1. A\n2. B"
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert "degree centrality" in result[0].lower()
        assert "test query" in result[0]
        assert "1. A\n2. B" in result[0]

    async def test_empty_context_returns_empty_list(self):
        retriever = CentralityRetriever()
        assert await retriever.get_completion_from_context(context=None) == []
        assert await retriever.get_completion_from_context(context="") == []


# ===== full retriever with mocked projection ===============================


@pytest.mark.asyncio
class TestCentralityRetrieverIntegration:
    async def test_get_retrieved_objects_entity_type_excluded(self, monkeypatch):
        async def mock_get_memory_fragment(**kwargs):
            return _build_graph_with_entity_type()

        monkeypatch.setattr(
            "cognee.modules.retrieval.centrality_retriever.get_memory_fragment",
            mock_get_memory_fragment,
        )

        retriever = CentralityRetriever(top_k=10)
        results = await retriever.get_retrieved_objects()

        node_ids = [r["node_id"] for r in results]
        assert "Service" not in node_ids, "EntityType node should be excluded"
        assert "A" in node_ids
        assert "B" in node_ids

    async def test_get_retrieved_objects_empty_graph(self, monkeypatch):
        async def mock_empty(*args, **kwargs):
            return CogneeGraph()

        monkeypatch.setattr(
            "cognee.modules.retrieval.centrality_retriever.get_memory_fragment",
            mock_empty,
        )

        retriever = CentralityRetriever()
        results = await retriever.get_retrieved_objects()
        assert results == []

    async def test_degree_mode_returns_top_k(self, monkeypatch):
        async def mock_fragment(**kwargs):
            return _build_test_graph()

        monkeypatch.setattr(
            "cognee.modules.retrieval.centrality_retriever.get_memory_fragment",
            mock_fragment,
        )

        retriever = CentralityRetriever(top_k=2, mode="degree")
        results = await retriever.get_retrieved_objects()
        assert len(results) <= 2
        assert all(r["score"] >= 1.0 for r in results)

    async def test_pagerank_mode_scores_sum_to_one(self, monkeypatch):
        async def mock_fragment(**kwargs):
            return _build_test_graph()

        monkeypatch.setattr(
            "cognee.modules.retrieval.centrality_retriever.get_memory_fragment",
            mock_fragment,
        )

        retriever = CentralityRetriever(top_k=10, mode="pagerank")
        results = await retriever.get_retrieved_objects()
        total = sum(r["score"] for r in results)
        assert abs(total - 1.0) < 0.01
