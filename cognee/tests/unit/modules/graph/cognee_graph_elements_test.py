import numpy as np
import pytest

from cognee.exceptions import InvalidValueError
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node


def test_node_initialization():
    """Test that a Node is initialized correctly."""
    node = Node("node1", {"attr1": "value1"}, dimension=2)
    assert node.id == "node1"
    assert node.attributes == {"attr1": "value1", "vector_distance": np.inf}
    assert len(node.status) == 2
    assert np.all(node.status == 1)


def test_node_invalid_dimension():
    """Test that initializing a Node with a non-positive dimension raises an error."""
    with pytest.raises(InvalidValueError, match="Dimension must be a positive integer"):
        Node("node1", dimension=0)


def test_add_skeleton_neighbor():
    """Test adding a neighbor to a node."""
    node1 = Node("node1")
    node2 = Node("node2")
    node1.add_skeleton_neighbor(node2)
    assert node2 in node1.skeleton_neighbours


def test_remove_skeleton_neighbor():
    """Test removing a neighbor from a node."""
    node1 = Node("node1")
    node2 = Node("node2")
    node1.add_skeleton_neighbor(node2)
    node1.remove_skeleton_neighbor(node2)
    assert node2 not in node1.skeleton_neighbours


def test_add_skeleton_edge():
    """Test adding an edge updates both skeleton_edges and skeleton_neighbours."""
    node1 = Node("node1")
    node2 = Node("node2")
    edge = Edge(node1, node2)
    node1.add_skeleton_edge(edge)
    assert edge in node1.skeleton_edges
    assert node2 in node1.skeleton_neighbours


def test_remove_skeleton_edge():
    """Test removing an edge updates both skeleton_edges and skeleton_neighbours."""
    node1 = Node("node1")
    node2 = Node("node2")
    edge = Edge(node1, node2)
    node1.add_skeleton_edge(edge)
    node1.remove_skeleton_edge(edge)
    assert edge not in node1.skeleton_edges
    assert node2 not in node1.skeleton_neighbours


def test_is_node_alive_in_dimension():
    """Test checking node's alive status in a specific dimension."""
    node = Node("node1", dimension=2)
    assert node.is_node_alive_in_dimension(1)
    node.status[1] = 0
    assert not node.is_node_alive_in_dimension(1)


def test_node_alive_invalid_dimension():
    """Test that checking alive status with an invalid dimension raises an error."""
    node = Node("node1", dimension=1)
    with pytest.raises(InvalidValueError, match="Dimension 1 is out of range"):
        node.is_node_alive_in_dimension(1)


def test_node_equality():
    """Test equality between nodes."""
    node1 = Node("node1")
    node2 = Node("node1")
    assert node1 == node2


def test_node_hash():
    """Test hashing for Node."""
    node = Node("node1")
    assert hash(node) == hash("node1")


### Tests for Edge ###


def test_edge_initialization():
    """Test that an Edge is initialized correctly."""
    node1 = Node("node1")
    node2 = Node("node2")
    edge = Edge(node1, node2, {"weight": 10}, directed=False, dimension=2)
    assert edge.node1 == node1
    assert edge.node2 == node2
    assert edge.attributes == {"vector_distance": np.inf, "weight": 10}
    assert edge.directed is False
    assert len(edge.status) == 2
    assert np.all(edge.status == 1)


def test_edge_invalid_dimension():
    """Test that initializing an Edge with a non-positive dimension raises an error."""
    node1 = Node("node1")
    node2 = Node("node2")
    with pytest.raises(InvalidValueError, match="Dimensions must be a positive integer."):
        Edge(node1, node2, dimension=0)


def test_is_edge_alive_in_dimension():
    """Test checking edge's alive status in a specific dimension."""
    node1 = Node("node1")
    node2 = Node("node2")
    edge = Edge(node1, node2, dimension=2)
    assert edge.is_edge_alive_in_dimension(1)
    edge.status[1] = 0
    assert not edge.is_edge_alive_in_dimension(1)


def test_edge_alive_invalid_dimension():
    """Test that checking alive status with an invalid dimension raises an error."""
    node1 = Node("node1")
    node2 = Node("node2")
    edge = Edge(node1, node2, dimension=1)
    with pytest.raises(InvalidValueError, match="Dimension 1 is out of range"):
        edge.is_edge_alive_in_dimension(1)


def test_edge_equality_directed():
    """Test equality between directed edges."""
    node1 = Node("node1")
    node2 = Node("node2")
    edge1 = Edge(node1, node2, directed=True)
    edge2 = Edge(node1, node2, directed=True)
    assert edge1 == edge2


def test_edge_equality_undirected():
    """Test equality between undirected edges."""
    node1 = Node("node1")
    node2 = Node("node2")
    edge1 = Edge(node1, node2, directed=False)
    edge2 = Edge(node2, node1, directed=False)
    assert edge1 == edge2


def test_edge_hash_directed():
    """Test hashing for directed edges."""
    node1 = Node("node1")
    node2 = Node("node2")
    edge = Edge(node1, node2, directed=True)
    assert hash(edge) == hash((node1, node2))


def test_edge_hash_undirected():
    """Test hashing for undirected edges."""
    node1 = Node("node1")
    node2 = Node("node2")
    edge = Edge(node1, node2, directed=False)
    assert hash(edge) == hash(frozenset({node1, node2}))
