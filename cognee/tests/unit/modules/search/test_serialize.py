"""Tests for search result serialization (circular reference handling)."""

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.search.methods._serialize import serialize_result_objects


def test_serialize_edge_breaks_circular_ref():
    """Edge→Node→skeleton_edges→Edge cycle should not cause recursion."""
    node_a = Node("a", attributes={"name": "Alice"})
    node_b = Node("b", attributes={"name": "Bob"})
    edge = Edge(node_a, node_b, attributes={"relation": "knows"})
    node_a.add_skeleton_edge(edge)
    node_b.add_skeleton_edge(edge)

    result = serialize_result_objects([edge])
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["source_node_id"] == "a"
    assert result[0]["target_node_id"] == "b"
    assert result[0]["edge_attributes"] == {"relation": "knows"}


def test_serialize_node():
    node = Node("x", attributes={"val": 1})
    result = serialize_result_objects([node])
    assert result[0]["id"] == "x"
    assert result[0]["attributes"] == {"val": 1}


def test_serialize_none():
    assert serialize_result_objects(None) is None


def test_serialize_non_graph_objects():
    result = serialize_result_objects(["hello", 42])
    assert result == ["hello", 42]
