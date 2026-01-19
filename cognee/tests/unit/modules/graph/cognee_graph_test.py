import pytest
from unittest.mock import AsyncMock

from cognee.modules.engine.utils.generate_edge_id import generate_edge_id
from cognee.modules.graph.exceptions import EntityNotFoundError, EntityAlreadyExistsError
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node


@pytest.fixture
def setup_graph():
    """Fixture to initialize a CogneeGraph instance."""
    return CogneeGraph()


@pytest.fixture
def mock_adapter():
    """Fixture to create a mock adapter for database operations."""
    adapter = AsyncMock()
    return adapter


@pytest.fixture
def mock_vector_engine():
    """Fixture to create a mock vector engine."""
    engine = AsyncMock()
    engine.search = AsyncMock()
    return engine


class MockScoredResult:
    """Mock class for vector search results."""

    def __init__(self, id, score, payload=None):
        self.id = id
        self.score = score
        self.payload = payload or {}


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


@pytest.mark.asyncio
async def test_project_graph_from_db_full_graph(setup_graph, mock_adapter):
    """Test projecting a full graph from database."""
    graph = setup_graph

    nodes_data = [
        ("1", {"name": "Node1", "description": "First node"}),
        ("2", {"name": "Node2", "description": "Second node"}),
    ]
    edges_data = [
        ("1", "2", "CONNECTS_TO", {"relationship_name": "connects"}),
    ]

    mock_adapter.get_graph_data = AsyncMock(return_value=(nodes_data, edges_data))

    await graph.project_graph_from_db(
        adapter=mock_adapter,
        node_properties_to_project=["name", "description"],
        edge_properties_to_project=["relationship_name"],
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.get_node("1") is not None
    assert graph.get_node("2") is not None
    assert graph.edges[0].node1.id == "1"
    assert graph.edges[0].node2.id == "2"


@pytest.mark.asyncio
async def test_project_graph_from_db_id_filtered(setup_graph, mock_adapter):
    """Test projecting an ID-filtered graph from database."""
    graph = setup_graph

    nodes_data = [
        ("1", {"name": "Node1"}),
        ("2", {"name": "Node2"}),
    ]
    edges_data = [
        ("1", "2", "CONNECTS_TO", {"relationship_name": "connects"}),
    ]

    mock_adapter.get_id_filtered_graph_data = AsyncMock(return_value=(nodes_data, edges_data))

    await graph.project_graph_from_db(
        adapter=mock_adapter,
        node_properties_to_project=["name"],
        edge_properties_to_project=["relationship_name"],
        relevant_ids_to_filter=["1", "2"],
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    mock_adapter.get_id_filtered_graph_data.assert_called_once()


@pytest.mark.asyncio
async def test_project_graph_from_db_nodeset_subgraph(setup_graph, mock_adapter):
    """Test projecting a nodeset subgraph filtered by node type and name."""
    graph = setup_graph

    nodes_data = [
        ("1", {"name": "Alice", "type": "Person"}),
        ("2", {"name": "Bob", "type": "Person"}),
    ]
    edges_data = [
        ("1", "2", "KNOWS", {"relationship_name": "knows"}),
    ]

    mock_adapter.get_nodeset_subgraph = AsyncMock(return_value=(nodes_data, edges_data))

    await graph.project_graph_from_db(
        adapter=mock_adapter,
        node_properties_to_project=["name", "type"],
        edge_properties_to_project=["relationship_name"],
        node_type="Person",
        node_name=["Alice"],
    )

    assert len(graph.nodes) == 2
    assert graph.get_node("1") is not None
    assert len(graph.edges) == 1
    mock_adapter.get_nodeset_subgraph.assert_called_once()


@pytest.mark.asyncio
async def test_project_graph_from_db_empty_graph(setup_graph, mock_adapter):
    """Test projecting empty graph raises EntityNotFoundError."""
    graph = setup_graph

    mock_adapter.get_graph_data = AsyncMock(return_value=([], []))

    with pytest.raises(EntityNotFoundError, match="Empty graph projected from the database."):
        await graph.project_graph_from_db(
            adapter=mock_adapter,
            node_properties_to_project=["name"],
            edge_properties_to_project=[],
        )


@pytest.mark.asyncio
async def test_project_graph_from_db_stores_triplet_penalty_on_graph(mock_adapter):
    """Test that project_graph_from_db stores triplet_distance_penalty on the graph."""
    from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph

    nodes_data = [("1", {"name": "Node1"})]
    edges_data = [("1", "1", "SELF", {})]

    mock_adapter.get_graph_data = AsyncMock(return_value=(nodes_data, edges_data))

    graph = CogneeGraph()
    custom_penalty = 5.0
    await graph.project_graph_from_db(
        adapter=mock_adapter,
        node_properties_to_project=["name"],
        edge_properties_to_project=[],
        triplet_distance_penalty=custom_penalty,
    )

    assert graph.triplet_distance_penalty == custom_penalty

    graph2 = CogneeGraph()
    await graph2.project_graph_from_db(
        adapter=mock_adapter,
        node_properties_to_project=["name"],
        edge_properties_to_project=[],
    )

    assert graph2.triplet_distance_penalty == 3.5


@pytest.mark.asyncio
async def test_project_graph_from_db_missing_nodes(setup_graph, mock_adapter):
    """Test that edges referencing missing nodes raise error."""
    graph = setup_graph

    nodes_data = [
        ("1", {"name": "Node1"}),
    ]
    edges_data = [
        ("1", "999", "CONNECTS_TO", {"relationship_name": "connects"}),
    ]

    mock_adapter.get_graph_data = AsyncMock(return_value=(nodes_data, edges_data))

    with pytest.raises(EntityNotFoundError, match="Edge references nonexistent nodes"):
        await graph.project_graph_from_db(
            adapter=mock_adapter,
            node_properties_to_project=["name"],
            edge_properties_to_project=["relationship_name"],
        )


@pytest.mark.asyncio
async def test_map_vector_distances_to_graph_nodes(setup_graph):
    """Test mapping vector distances to graph nodes."""
    graph = setup_graph

    node1 = Node("1", {"name": "Node1"})
    node2 = Node("2", {"name": "Node2"})
    graph.add_node(node1)
    graph.add_node(node2)

    node_distances = {
        "Entity_name": [
            MockScoredResult("1", 0.95),
            MockScoredResult("2", 0.87),
        ]
    }

    await graph.map_vector_distances_to_graph_nodes(node_distances)

    assert graph.get_node("1").attributes.get("vector_distance") == [0.95]
    assert graph.get_node("2").attributes.get("vector_distance") == [0.87]


@pytest.mark.asyncio
async def test_map_vector_distances_partial_node_coverage(setup_graph):
    """Test mapping vector distances when only some nodes have results."""
    graph = setup_graph

    node1 = Node("1", {"name": "Node1"})
    node2 = Node("2", {"name": "Node2"})
    node3 = Node("3", {"name": "Node3"})
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    node_distances = {
        "Entity_name": [
            MockScoredResult("1", 0.95),
            MockScoredResult("2", 0.87),
        ]
    }

    await graph.map_vector_distances_to_graph_nodes(node_distances)

    assert graph.get_node("1").attributes.get("vector_distance") == [0.95]
    assert graph.get_node("2").attributes.get("vector_distance") == [0.87]
    assert graph.get_node("3").attributes.get("vector_distance") == [3.5]


@pytest.mark.asyncio
async def test_map_vector_distances_multiple_categories(setup_graph):
    """Test mapping vector distances from multiple collection categories."""
    graph = setup_graph

    # Create nodes
    node1 = Node("1")
    node2 = Node("2")
    node3 = Node("3")
    node4 = Node("4")
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)

    node_distances = {
        "Entity_name": [
            MockScoredResult("1", 0.95),
            MockScoredResult("2", 0.87),
        ],
        "TextSummary_text": [
            MockScoredResult("3", 0.92),
        ],
    }

    await graph.map_vector_distances_to_graph_nodes(node_distances)

    assert graph.get_node("1").attributes.get("vector_distance") == [0.95]
    assert graph.get_node("2").attributes.get("vector_distance") == [0.87]
    assert graph.get_node("3").attributes.get("vector_distance") == [0.92]
    assert graph.get_node("4").attributes.get("vector_distance") == [3.5]


@pytest.mark.asyncio
async def test_map_vector_distances_to_graph_nodes_multi_query(setup_graph):
    """Test mapping vector distances with multiple queries."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    node3 = Node("3")
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    node_distances = {
        "Entity_name": [
            [MockScoredResult("1", 0.95)],  # query 0
            [MockScoredResult("2", 0.87)],  # query 1
        ]
    }

    await graph.map_vector_distances_to_graph_nodes(node_distances, query_list_length=2)

    assert graph.get_node("1").attributes.get("vector_distance") == [0.95, 3.5]
    assert graph.get_node("2").attributes.get("vector_distance") == [3.5, 0.87]
    assert graph.get_node("3").attributes.get("vector_distance") == [3.5, 3.5]


@pytest.mark.asyncio
async def test_map_vector_distances_to_graph_edges_with_payload(setup_graph):
    """Test mapping vector distances to edges when edge_distances provided."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    graph.add_node(node1)
    graph.add_node(node2)

    edge = Edge(
        node1,
        node2,
        attributes={"edge_text": "CONNECTS_TO", "relationship_type": "connects"},
    )
    graph.add_edge(edge)

    edge_distances = [
        MockScoredResult(generate_edge_id("CONNECTS_TO"), 0.92, payload={"text": "CONNECTS_TO"}),
    ]

    await graph.map_vector_distances_to_graph_edges(edge_distances=edge_distances)

    assert graph.edges[0].attributes.get("vector_distance") == [0.92]


@pytest.mark.asyncio
async def test_map_vector_distances_partial_edge_coverage(setup_graph):
    """Test mapping edge distances when only some edges have results."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    node3 = Node("3")
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    edge1 = Edge(node1, node2, attributes={"edge_text": "CONNECTS_TO"})
    edge2 = Edge(node2, node3, attributes={"edge_text": "DEPENDS_ON"})
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    edge_1_text = "CONNECTS_TO"
    edge_distances = [
        MockScoredResult(generate_edge_id(edge_1_text), 0.92, payload={"text": edge_1_text}),
    ]

    await graph.map_vector_distances_to_graph_edges(edge_distances=edge_distances)

    assert graph.edges[0].attributes.get("vector_distance") == [0.92]
    assert graph.edges[1].attributes.get("vector_distance") == [3.5]


@pytest.mark.asyncio
async def test_map_vector_distances_edges_fallback_to_relationship_type(setup_graph):
    """Test that edge mapping falls back to relationship_type when edge_text is missing."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    graph.add_node(node1)
    graph.add_node(node2)

    edge = Edge(
        node1,
        node2,
        attributes={"relationship_type": "KNOWS"},
    )
    graph.add_edge(edge)

    edge_text = "KNOWS"
    edge_distances = [
        MockScoredResult(generate_edge_id(edge_text), 0.85, payload={"text": edge_text}),
    ]

    await graph.map_vector_distances_to_graph_edges(edge_distances=edge_distances)

    assert graph.edges[0].attributes.get("vector_distance") == [0.85]


@pytest.mark.asyncio
async def test_map_vector_distances_no_edge_matches(setup_graph):
    """Test edge mapping when no edges match the distance results."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    graph.add_node(node1)
    graph.add_node(node2)

    edge = Edge(
        node1,
        node2,
        attributes={"edge_text": "CONNECTS_TO", "relationship_type": "connects"},
    )
    graph.add_edge(edge)

    edge_text = "SOME_OTHER_EDGE"
    edge_distances = [
        MockScoredResult(generate_edge_id(edge_text), 0.92, payload={"text": edge_text}),
    ]

    await graph.map_vector_distances_to_graph_edges(edge_distances=edge_distances)

    assert graph.edges[0].attributes.get("vector_distance") == [3.5]


@pytest.mark.asyncio
async def test_map_vector_distances_none_returns_early(setup_graph):
    """Test that edge_distances=None returns early without error and vector_distance is set to default penalty."""
    graph = setup_graph
    graph.add_node(Node("1"))
    graph.add_node(Node("2"))
    graph.add_edge(Edge(graph.get_node("1"), graph.get_node("2")))

    await graph.map_vector_distances_to_graph_edges(edge_distances=None)

    assert graph.edges[0].attributes.get("vector_distance") == [3.5]


@pytest.mark.asyncio
async def test_map_vector_distances_empty_nodes_returns_early(setup_graph):
    """Test that node_distances={} returns early without error and vector_distance is set to default penalty."""
    graph = setup_graph
    node1 = Node("1")
    node2 = Node("2")
    graph.add_node(node1)
    graph.add_node(node2)

    await graph.map_vector_distances_to_graph_nodes({})

    assert node1.attributes.get("vector_distance") == [3.5]
    assert node2.attributes.get("vector_distance") == [3.5]


@pytest.mark.asyncio
async def test_map_vector_distances_to_graph_edges_multi_query(setup_graph):
    """Test mapping edge distances with multiple queries."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    node3 = Node("3")
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    edge1 = Edge(node1, node2, attributes={"edge_text": "A"})
    edge2 = Edge(node2, node3, attributes={"edge_text": "B"})
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    edge_1_text = "A"
    edge_2_text = "B"
    edge_distances = [
        [
            MockScoredResult(generate_edge_id(edge_1_text), 0.1, payload={"text": edge_1_text})
        ],  # query 0
        [
            MockScoredResult(generate_edge_id(edge_2_text), 0.2, payload={"text": edge_2_text})
        ],  # query 1
    ]

    await graph.map_vector_distances_to_graph_edges(
        edge_distances=edge_distances, query_list_length=2
    )

    assert graph.edges[0].attributes.get("vector_distance") == [0.1, 3.5]
    assert graph.edges[1].attributes.get("vector_distance") == [3.5, 0.2]


@pytest.mark.asyncio
async def test_map_vector_distances_to_graph_edges_preserves_unmapped_indices(setup_graph):
    """Test that unmapped indices in multi-query mode stay at default penalty."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    node3 = Node("3")
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    edge1 = Edge(node1, node2, attributes={"edge_text": "A"})
    edge2 = Edge(node2, node3, attributes={"edge_text": "B"})
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    edge_1_text = "A"
    edge_distances = [
        [
            MockScoredResult(generate_edge_id(edge_1_text), 0.1, payload={"text": edge_1_text})
        ],  # query 0: only edge1 mapped
        [],  # query 1: no edges mapped
    ]

    await graph.map_vector_distances_to_graph_edges(
        edge_distances=edge_distances, query_list_length=2
    )

    assert graph.edges[0].attributes.get("vector_distance") == [0.1, 3.5]
    assert graph.edges[1].attributes.get("vector_distance") == [3.5, 3.5]


@pytest.mark.asyncio
async def test_calculate_top_triplet_importances(setup_graph):
    """Test calculating top triplet importances by score."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    node3 = Node("3")
    node4 = Node("4")

    node1.add_attribute("vector_distance", [0.9])
    node2.add_attribute("vector_distance", [0.8])
    node3.add_attribute("vector_distance", [0.7])
    node4.add_attribute("vector_distance", [0.6])

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)

    edge1 = Edge(node1, node2)
    edge2 = Edge(node2, node3)
    edge3 = Edge(node3, node4)

    edge1.add_attribute("vector_distance", [0.85])
    edge2.add_attribute("vector_distance", [0.75])
    edge3.add_attribute("vector_distance", [0.65])

    graph.add_edge(edge1)
    graph.add_edge(edge2)
    graph.add_edge(edge3)

    top_triplets = await graph.calculate_top_triplet_importances(k=2)

    assert len(top_triplets) == 2

    assert top_triplets[0] == edge3
    assert top_triplets[1] == edge2


@pytest.mark.asyncio
async def test_calculate_top_triplet_importances_default_distances(setup_graph):
    """Test that vector_distance stays None when no distances are passed and calculate_top_triplet_importances handles it."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    graph.add_node(node1)
    graph.add_node(node2)

    edge = Edge(node1, node2)
    graph.add_edge(edge)

    # Verify vector_distance is None when no distances are passed
    assert node1.attributes.get("vector_distance") is None
    assert node2.attributes.get("vector_distance") is None
    assert edge.attributes.get("vector_distance") is None

    # When no distances are set, calculate_top_triplet_importances should handle None
    # by either raising an error or skipping edges with None distances
    with pytest.raises(ValueError):
        await graph.calculate_top_triplet_importances(k=1)


@pytest.mark.asyncio
async def test_calculate_top_triplet_importances_single_query_via_helper(setup_graph):
    """Test calculating top triplet importances for a single query index."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    node3 = Node("3")
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    node1.add_attribute("vector_distance", [0.1])
    node2.add_attribute("vector_distance", [0.2])
    node3.add_attribute("vector_distance", [0.3])

    edge1 = Edge(node1, node2)
    edge2 = Edge(node2, node3)
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    edge1.add_attribute("vector_distance", [0.3])
    edge2.add_attribute("vector_distance", [0.4])

    results = await graph.calculate_top_triplet_importances(k=1, query_list_length=1)
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0] == edge1


@pytest.mark.asyncio
async def test_calculate_top_triplet_importances_multi_query(setup_graph):
    """Test calculating top triplet importances with multiple queries."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    node3 = Node("3")
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    edge_a = Edge(node1, node2)
    edge_b = Edge(node2, node3)
    graph.add_edge(edge_a)
    graph.add_edge(edge_b)

    node1.add_attribute("vector_distance", [0.1, 0.9])
    node2.add_attribute("vector_distance", [0.1, 0.9])
    node3.add_attribute("vector_distance", [0.9, 0.1])
    edge_a.add_attribute("vector_distance", [0.1, 0.9])
    edge_b.add_attribute("vector_distance", [0.9, 0.1])

    results = await graph.calculate_top_triplet_importances(k=1, query_list_length=2)

    assert len(results) == 2
    assert results[0][0] == edge_a
    assert results[1][0] == edge_b


@pytest.mark.asyncio
async def test_calculate_top_triplet_importances_raises_on_short_list(setup_graph):
    """Test that scoring raises ValueError when list is too short for query_index."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    graph.add_node(node1)
    graph.add_node(node2)

    node1.add_attribute("vector_distance", [0.1])
    node2.add_attribute("vector_distance", [0.2])

    edge = Edge(node1, node2)
    edge.add_attribute("vector_distance", [0.3])
    graph.add_edge(edge)

    with pytest.raises(ValueError):
        await graph.calculate_top_triplet_importances(k=1, query_list_length=2)


@pytest.mark.asyncio
async def test_calculate_top_triplet_importances_raises_on_missing_attribute(setup_graph):
    """Test that scoring raises error when vector_distance is missing."""
    graph = setup_graph

    node1 = Node("1")
    node2 = Node("2")
    graph.add_node(node1)
    graph.add_node(node2)

    del node1.attributes["vector_distance"]
    del node2.attributes["vector_distance"]

    edge = Edge(node1, node2)
    del edge.attributes["vector_distance"]
    graph.add_edge(edge)

    with pytest.raises(ValueError):
        await graph.calculate_top_triplet_importances(k=1, query_list_length=1)


def test_normalize_query_distance_lists_flat_list_single_query(setup_graph):
    """Test that flat list is normalized to list-of-lists with length 1 for single-query mode."""
    graph = setup_graph
    flat_list = [MockScoredResult("node1", 0.95), MockScoredResult("node2", 0.87)]

    result = graph._normalize_query_distance_lists(flat_list, query_list_length=None, name="test")

    assert len(result) == 1
    assert result[0] == flat_list


def test_normalize_query_distance_lists_nested_list_batch_mode(setup_graph):
    """Test that nested list is used as-is when query_list_length matches."""
    graph = setup_graph
    nested_list = [
        [MockScoredResult("node1", 0.95)],
        [MockScoredResult("node2", 0.87)],
    ]

    result = graph._normalize_query_distance_lists(nested_list, query_list_length=2, name="test")

    assert len(result) == 2
    assert result == nested_list


def test_normalize_query_distance_lists_raises_on_length_mismatch(setup_graph):
    """Test that ValueError is raised when nested list length doesn't match query_list_length."""
    graph = setup_graph
    nested_list = [
        [MockScoredResult("node1", 0.95)],
        [MockScoredResult("node2", 0.87)],
    ]

    with pytest.raises(ValueError, match="test has 2 query lists, but query_list_length is 3"):
        graph._normalize_query_distance_lists(nested_list, query_list_length=3, name="test")


def test_normalize_query_distance_lists_empty_list(setup_graph):
    """Test that empty list returns empty list."""
    graph = setup_graph

    result = graph._normalize_query_distance_lists([], query_list_length=None, name="test")

    assert result == []
