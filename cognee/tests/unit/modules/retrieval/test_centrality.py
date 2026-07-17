"""Unit tests for pure-Python graph centrality (issue #3378).

Builds known ``CogneeGraph`` fragments by hand and asserts the structural
rankings, with no database, vector store, or LLM involved.
"""

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.utils.centrality import (
    degree_centrality,
    pagerank,
    rank_top_k,
    select_rankable_node_ids,
)


def _node(node_id, node_type="Entity", name=None):
    return Node(node_id, {"type": node_type, "name": name or node_id})


def _hub_graph():
    """Directed in-star: four leaves point at hub ``c``; every entity is_a ``t``.

    ``t`` is an EntityType taxonomy-label node that should be excluded from
    ranking even though every entity links to it.
    """
    graph = CogneeGraph()
    hub = _node("c")
    leaves = [_node(f"l{i}") for i in range(1, 5)]
    type_node = _node("t", node_type="EntityType", name="Thing")
    for node in [hub, *leaves, type_node]:
        graph.add_node(node)

    for leaf in leaves:
        graph.add_edge(
            Edge(leaf, hub, attributes={"relationship_type": "points_to"}, directed=True)
        )
    for entity in [hub, *leaves]:
        graph.add_edge(
            Edge(entity, type_node, attributes={"relationship_type": "is_a"}, directed=True)
        )
    return graph


def test_select_rankable_node_ids_drops_entity_type_nodes():
    graph = _hub_graph()
    rankable = select_rankable_node_ids(graph)
    assert "t" not in rankable
    assert set(rankable) == {"c", "l1", "l2", "l3", "l4"}


def test_degree_centrality_ranks_hub_first():
    graph = _hub_graph()
    rankable = select_rankable_node_ids(graph)
    scores = degree_centrality(graph, rankable)

    # Hub neighbors all four leaves -> normalized degree 1.0; leaves only touch c.
    assert scores["c"] == 1.0
    assert all(scores[f"l{i}"] == 0.25 for i in range(1, 5))
    assert rank_top_k(scores, 1) == ["c"]


def test_pagerank_ranks_hub_first_and_is_normalized():
    graph = _hub_graph()
    rankable = select_rankable_node_ids(graph)
    scores = pagerank(graph, rankable)

    assert rank_top_k(scores, 1) == ["c"]
    assert all(scores["c"] > scores[f"l{i}"] for i in range(1, 5))
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_pagerank_personalization_biases_toward_seed():
    graph = _hub_graph()
    rankable = select_rankable_node_ids(graph)

    uniform = pagerank(graph, rankable)
    personalized = pagerank(graph, rankable, personalization={"l1": 1.0})

    # Teleporting onto l1 must raise its score relative to the unbiased run.
    assert personalized["l1"] > uniform["l1"]
    assert abs(sum(personalized.values()) - 1.0) < 1e-6


def test_rank_top_k_is_deterministic_on_ties():
    # Equal scores must break ties by id so output is reproducible.
    scores = {"b": 0.5, "a": 0.5, "c": 0.1}
    assert rank_top_k(scores, 2) == ["a", "b"]


def test_centrality_handles_trivial_graphs():
    empty = CogneeGraph()
    assert pagerank(empty, []) == {}
    assert degree_centrality(empty, []) == {}

    single = CogneeGraph()
    single.add_node(_node("only"))
    assert pagerank(single, ["only"]) == {"only": 1.0}
    assert degree_centrality(single, ["only"]) == {"only": 0.0}
