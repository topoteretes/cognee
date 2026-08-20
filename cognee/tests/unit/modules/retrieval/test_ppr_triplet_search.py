"""Unit tests for Personalized-PageRank triplet search.

The headline test (`test_ppr_surfaces_bridge_over_leaf`) proves the point of the
whole retriever: a fact that bridges two query-relevant entities ranks above a
leaf fact, even though neither the bridge nor the leaf has any direct vector
similarity to the query. A vector-only ranking cannot tell them apart; PPR can.

All tests are offline: the vector search and the graph projection are mocked, so
the graph structure and the vector hits are fully controlled and no database or
embedding model is touched.
"""

import pytest
from unittest.mock import AsyncMock, patch

from cognee.exceptions import CogneeValidationError
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.utils.ppr_triplet_search import ppr_triplet_search

PPR_MODULE = "cognee.modules.retrieval.utils.ppr_triplet_search"


class _MockScored:
    """Minimal stand-in for a vector search result (has id and score)."""

    def __init__(self, id, score):
        self.id = id
        self.score = score


class _FakeVectorSearch:
    """Fake NodeEdgeVectorSearch: skips embedding/DB and exposes controlled hits."""

    def __init__(self, node_distances, has_results=True, relevant_ids=None):
        self.node_distances = node_distances
        self.edge_distances = []
        self.query_vector = [0.1, 0.2, 0.3]
        self.query_list_length = None
        self._has_results = has_results
        self._relevant_ids = relevant_ids or []

    async def embed_and_retrieve_distances(self, *args, **kwargs):
        return None

    def has_results(self):
        return self._has_results

    def extract_relevant_node_ids(self):
        return list(self._relevant_ids)


def _node(node_id):
    return Node(node_id, {"name": node_id})


def _bridge_graph():
    """S1 and S2 are query-relevant seeds. C bridges them (connected to both);
    P is a leaf hanging off S1 only. Neither C nor P has a vector match."""
    graph = CogneeGraph()
    for node_id in ("S1", "S2", "C", "P"):
        graph.add_node(_node(node_id))
    n = graph.get_node
    graph.add_edge(Edge(n("S1"), n("C")))  # ("S1","C")
    graph.add_edge(Edge(n("C"), n("S2")))  # ("C","S2")
    graph.add_edge(Edge(n("S1"), n("P")))  # ("S1","P")  <- leaf
    return graph


def _edge_pairs(edges):
    return {(e.node1.id, e.node2.id) for e in edges}


@pytest.mark.asyncio
async def test_ppr_surfaces_bridge_over_leaf():
    """The two bridge triplets outrank the leaf triplet, even though the bridge
    node C has no direct vector similarity. Vector-only ranking would tie them."""
    graph = _bridge_graph()
    # Only S1 and S2 matched the query (low cosine distance = high similarity).
    fake_vs = _FakeVectorSearch(
        node_distances={"Entity_name": [_MockScored("S1", 0.1), _MockScored("S2", 0.1)]},
        relevant_ids=["S1", "S2"],
    )

    with (
        patch(f"{PPR_MODULE}.NodeEdgeVectorSearch", return_value=fake_vs),
        patch(f"{PPR_MODULE}.get_memory_fragment", new=AsyncMock(return_value=graph)),
    ):
        results = await ppr_triplet_search("how are S1 and S2 related", top_k=2, ppr_weight=1.0)

    pairs = _edge_pairs(results)
    assert ("S1", "P") not in pairs, f"leaf triplet should be ranked last, got {pairs}"
    assert pairs == {("S1", "C"), ("C", "S2")}, f"expected the two bridge triplets, got {pairs}"


@pytest.mark.asyncio
async def test_ppr_ranks_bridge_node_above_leaf_node():
    """Directly: the bridge node C accrues more PageRank mass than the leaf P, so
    the S1-C triplet outranks the S1-P triplet (they share S1, so the ordering is
    decided purely by C vs P)."""
    graph = _bridge_graph()
    fake_vs = _FakeVectorSearch(
        node_distances={"Entity_name": [_MockScored("S1", 0.1), _MockScored("S2", 0.1)]},
        relevant_ids=["S1", "S2"],
    )

    with (
        patch(f"{PPR_MODULE}.NodeEdgeVectorSearch", return_value=fake_vs),
        patch(f"{PPR_MODULE}.get_memory_fragment", new=AsyncMock(return_value=graph)),
    ):
        results = await ppr_triplet_search("how are S1 and S2 related", top_k=1, ppr_weight=1.0)

    assert _edge_pairs(results) != {("S1", "P")}
    assert results[0].node1.id == "S1" and results[0].node2.id == "C"


@pytest.mark.asyncio
async def test_returns_empty_when_no_vector_results():
    """No vector hits -> empty result, without touching the graph projection."""
    fake_vs = _FakeVectorSearch(node_distances={}, has_results=False)
    with patch(f"{PPR_MODULE}.NodeEdgeVectorSearch", return_value=fake_vs):
        results = await ppr_triplet_search("anything", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_returns_empty_when_fragment_has_no_edges():
    """A fragment with nodes but no edges yields no triplets."""
    graph = CogneeGraph()
    graph.add_node(_node("S1"))
    fake_vs = _FakeVectorSearch(
        node_distances={"Entity_name": [_MockScored("S1", 0.1)]}, relevant_ids=["S1"]
    )
    with (
        patch(f"{PPR_MODULE}.NodeEdgeVectorSearch", return_value=fake_vs),
        patch(f"{PPR_MODULE}.get_memory_fragment", new=AsyncMock(return_value=graph)),
    ):
        results = await ppr_triplet_search("q", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_respects_top_k():
    """Never returns more than top_k triplets."""
    graph = _bridge_graph()
    fake_vs = _FakeVectorSearch(
        node_distances={"Entity_name": [_MockScored("S1", 0.1), _MockScored("S2", 0.1)]},
        relevant_ids=["S1", "S2"],
    )
    with (
        patch(f"{PPR_MODULE}.NodeEdgeVectorSearch", return_value=fake_vs),
        patch(f"{PPR_MODULE}.get_memory_fragment", new=AsyncMock(return_value=graph)),
    ):
        results = await ppr_triplet_search("q", top_k=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_survives_pagerank_failure_and_falls_back_to_similarity():
    """If PageRank raises, ranking degrades to similarity-only instead of erroring."""
    graph = _bridge_graph()
    fake_vs = _FakeVectorSearch(
        node_distances={"Entity_name": [_MockScored("S1", 0.1), _MockScored("S2", 0.1)]},
        relevant_ids=["S1", "S2"],
    )
    with (
        patch(f"{PPR_MODULE}.NodeEdgeVectorSearch", return_value=fake_vs),
        patch(f"{PPR_MODULE}.get_memory_fragment", new=AsyncMock(return_value=graph)),
        patch(f"{PPR_MODULE}.nx.pagerank", side_effect=RuntimeError("power iteration failed")),
    ):
        results = await ppr_triplet_search("q", top_k=3, ppr_weight=1.0)
    # Still returns triplets (does not raise); with ppr unavailable, similarity ranks them.
    assert len(results) == 3


@pytest.mark.asyncio
async def test_validation_errors():
    with pytest.raises(ValueError, match="non-empty string"):
        await ppr_triplet_search("", top_k=5)
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        await ppr_triplet_search("q", top_k=0)
    with pytest.raises(CogneeValidationError, match=r"feedback_influence must be in range"):
        await ppr_triplet_search("q", top_k=5, feedback_influence=1.5)


@pytest.mark.asyncio
async def test_dispatch_returns_ppr_retriever():
    """The GRAPH_COMPLETION_PPR search type resolves to the PPR retriever."""
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod
    from cognee.modules.retrieval.graph_completion_ppr_retriever import (
        GraphCompletionPPRRetriever,
    )
    from cognee.modules.search.types import SearchType

    retriever = await mod.get_search_type_retriever_instance(
        SearchType.GRAPH_COMPLETION_PPR,
        query_text="q",
        feedback_influence=0.4,
        retriever_specific_config={"ppr_alpha": 0.7, "ppr_weight": 0.9},
    )
    assert isinstance(retriever, GraphCompletionPPRRetriever)
    assert retriever.feedback_influence == 0.4
    assert retriever.ppr_alpha == 0.7
    assert retriever.ppr_weight == 0.9
    assert retriever.neighborhood_depth == 1
