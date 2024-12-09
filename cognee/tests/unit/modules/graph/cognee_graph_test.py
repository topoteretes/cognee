import pytest

from cognee.modules.graph.exceptions import EntityNotFoundError, EntityAlreadyExistsError
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node


@pytest.fixture
def setup_graph():
    """Fixture to initialize a CogneeGraph instance."""
    return CogneeGraph()


def test_add_node_success(setup_graph):
    """Test successful addition of a node."""
    graph = setup_graph
    node = Node("node1")
    graph.add_node(node)
    assert graph.get_node("node1") == node


def test_add_duplicate_node(setup_graph):
    """Test adding a duplicate node raises an exception."""
    graph = setup_graph
    node = Node("node1")
    graph.add_node(node)
    with pytest.raises(EntityAlreadyExistsError, match="Node with id node1 already exists."):
        graph.add_node(node)


def test_add_edge_success(setup_graph):
    """Test successful addition of an edge."""
    graph = setup_graph
    node1 = Node("node1")
    node2 = Node("node2")
    graph.add_node(node1)
    graph.add_node(node2)
    edge = Edge(node1, node2)
    graph.add_edge(edge)
    assert edge in graph.edges
    assert edge in node1.skeleton_edges
    assert edge in node2.skeleton_edges


def test_get_node_success(setup_graph):
    """Test retrieving an existing node."""
    graph = setup_graph
    node = Node("node1")
    graph.add_node(node)
    assert graph.get_node("node1") == node


def test_get_node_nonexistent(setup_graph):
    """Test retrieving a nonexistent node returns None."""
    graph = setup_graph
    assert graph.get_node("nonexistent") is None


def test_get_edges_success(setup_graph):
    """Test retrieving edges of a node."""
    graph = setup_graph
    node1 = Node("node1")
    node2 = Node("node2")
    graph.add_node(node1)
    graph.add_node(node2)
    edge = Edge(node1, node2)
    graph.add_edge(edge)
    assert edge in graph.get_edges_from_node("node1")


def test_get_edges_nonexistent_node(setup_graph):
    """Test retrieving edges for a nonexistent node raises an exception."""
    graph = setup_graph
    with pytest.raises(EntityNotFoundError, match="Node with id nonexistent does not exist."):
        graph.get_edges_from_node("nonexistent")
