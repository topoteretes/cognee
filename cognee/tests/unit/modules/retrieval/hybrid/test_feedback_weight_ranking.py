"""Learned feedback weights in the hybrid chunk lane, and the triplet-lane
importance/feedback ordering fix."""

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs


def _pair(chunk_id: str, rank: int) -> dict:
    return {
        "chunk": {"id": chunk_id, "payload": {"id": chunk_id}},
        "chunk_id": chunk_id,
        "bm25_rank": rank,
        "vector_rank": rank,
        "summary_rank": None,
    }


def test_feedback_weights_rerank_equal_candidates():
    """Same retrieval ranks; only the learned weight differs -> boosted chunk wins."""
    ranked = rank_chunk_summary_pairs(
        [_pair("low", 0), _pair("high", 0)],
        limit=2,
        use_importance_weight=False,
        feedback_weight_by_id={"high": 0.9, "low": 0.1},
    )

    assert [pair["chunk_id"] for pair in ranked] == ["high", "low"]


def test_neutral_weight_map_keeps_baseline_order():
    """An empty map exercises the factor path at the neutral 1.0 for every chunk,
    so ranking must match the no-map baseline exactly."""
    baseline = rank_chunk_summary_pairs(
        [_pair("a", 0), _pair("b", 1)],
        limit=2,
        use_importance_weight=False,
    )
    with_neutral_map = rank_chunk_summary_pairs(
        [_pair("a", 0), _pair("b", 1)],
        limit=2,
        use_importance_weight=False,
        feedback_weight_by_id={},
    )

    assert [p["chunk_id"] for p in baseline] == [p["chunk_id"] for p in with_neutral_map]


def test_missing_chunk_id_ranks_at_neutral_factor():
    """A chunk absent from the weight map is neutral (1.0), not penalized."""
    ranked = rank_chunk_summary_pairs(
        [_pair("unknown", 0), _pair("penalized", 0)],
        limit=2,
        use_importance_weight=False,
        feedback_weight_by_id={"penalized": 0.1},
    )

    assert [pair["chunk_id"] for pair in ranked] == ["unknown", "penalized"]


@pytest.mark.asyncio
async def test_triplet_feedback_blend_applies_before_importance_scaling():
    """Regression for the ordering bug: importance scaling maps distances into
    [0, 4], and the feedback blend used to gate on the SCALED value, silently
    skipping any element whose scaled distance exceeded 2. Raw distance 1.5 with
    importance 0.5 scaled to 2.25 and escaped blending entirely."""
    graph = CogneeGraph()

    node1 = Node("1", {"feedback_weight": 0.95, "importance_weight": 0.5})
    node2 = Node("2", {"importance_weight": 0.5})
    node3 = Node("3", {"feedback_weight": 0.05, "importance_weight": 0.5})
    for node in (node1, node2, node3):
        graph.add_node(node)

    # edge_bad is inserted FIRST: under the old (broken) gating both edges tied
    # and heapq's stable order returned edge_bad, so this test fails without the fix.
    edge_bad = Edge(node2, node3, attributes={"feedback_weight": 0.05})
    edge_good = Edge(node1, node2, attributes={"feedback_weight": 0.95})
    graph.add_edge(edge_bad)
    graph.add_edge(edge_good)

    for element in (node1, node2, node3, edge_good, edge_bad):
        element.add_attribute("vector_distance", [1.5])
        element.attributes.setdefault("importance_weight", 0.5)

    results = await graph.calculate_top_triplet_importances(k=2, feedback_influence=0.5)

    assert results[0] == edge_good
    assert results[1] == edge_bad
