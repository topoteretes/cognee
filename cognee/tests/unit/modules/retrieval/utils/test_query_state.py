from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.utils.query_state import QueryState


def _edge(relationship_name: str = "likes") -> Edge:
    return Edge(
        Node("node-a", {"name": "A"}),
        Node("node-b", {"name": "B"}),
        attributes={"relationship_name": relationship_name},
    )


def test_merge_triplets_deduplicates_fresh_logically_equal_edges():
    state = QueryState(triplets=[_edge()])
    prev_size = len(state.triplets)

    state.merge_triplets([_edge()])
    state.check_convergence(prev_size)

    assert len(state.triplets) == 1
    assert state.done is True


def test_merge_triplets_keeps_distinct_relationships_between_same_nodes():
    state = QueryState(triplets=[_edge("likes")])

    state.merge_triplets([_edge("knows")])

    assert len(state.triplets) == 2
