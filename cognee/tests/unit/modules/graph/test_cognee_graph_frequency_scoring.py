from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node


def _node(node_id: str, frequency_weight: float) -> Node:
    node = Node(
        node_id,
        {
            "importance_weight": 1.0,
            "feedback_weight": 0.5,
            "frequency_weight": frequency_weight,
        },
    )
    node.attributes["vector_distance"] = [1.0]
    return node


def _edge(label: str, frequency_weight: float) -> Edge:
    node1 = _node(f"{label}-source", frequency_weight)
    node2 = _node(f"{label}-target", frequency_weight)
    edge = Edge(
        node1,
        node2,
        {
            "importance_weight": 1.0,
            "feedback_weight": 0.5,
            "frequency_weight": frequency_weight,
        },
    )
    edge.attributes["vector_distance"] = [1.0]
    return edge


def test_frequency_weight_boosts_tied_triplet_scores():
    graph = CogneeGraph()
    low_frequency_edge = _edge("low", 0.0)
    high_frequency_edge = _edge("high", 4.0)
    graph.edges = [low_frequency_edge, high_frequency_edge]

    results = graph._calculate_query_top_triplet_importances(
        k=2,
        query_index=0,
        feedback_influence=1.0,
    )

    assert results[0] is high_frequency_edge
