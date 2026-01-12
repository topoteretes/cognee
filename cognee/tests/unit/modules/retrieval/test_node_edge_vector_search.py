import pytest
from unittest.mock import AsyncMock

from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError


class MockScoredResult:
    """Mock class for vector search results."""

    def __init__(self, id, score, payload=None):
        self.id = id
        self.score = score
        self.payload = payload or {}


@pytest.mark.asyncio
async def test_node_edge_vector_search_single_query_shape():
    """Test that single query mode produces flat lists (not list-of-lists)."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    node_results = [MockScoredResult("node1", 0.95), MockScoredResult("node2", 0.87)]
    edge_results = [MockScoredResult("edge1", 0.92)]

    def search_side_effect(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "EdgeType_relationship_name":
            return edge_results
        return node_results

    mock_vector_engine.search = AsyncMock(side_effect=search_side_effect)

    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    collections = ["Entity_name", "EdgeType_relationship_name"]

    await vector_search.embed_and_retrieve_distances(
        query="test query", query_batch=None, collections=collections, wide_search_limit=10
    )

    assert vector_search.query_list_length is None
    assert vector_search.edge_distances == edge_results
    assert vector_search.node_distances["Entity_name"] == node_results
    mock_vector_engine.embedding_engine.embed_text.assert_called_once_with(["test query"])


@pytest.mark.asyncio
async def test_node_edge_vector_search_batch_query_shape_and_empties():
    """Test that batch query mode produces list-of-lists with correct length and handles empty collections."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()

    query_batch = ["query a", "query b"]
    node_results_query_a = [MockScoredResult("node1", 0.95)]
    node_results_query_b = [MockScoredResult("node2", 0.87)]
    edge_results_query_a = [MockScoredResult("edge1", 0.92)]
    edge_results_query_b = []

    def batch_search_side_effect(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "EdgeType_relationship_name":
            return [edge_results_query_a, edge_results_query_b]
        elif collection_name == "Entity_name":
            return [node_results_query_a, node_results_query_b]
        elif collection_name == "MissingCollection":
            raise CollectionNotFoundError("Collection not found")
        return [[], []]

    mock_vector_engine.batch_search = AsyncMock(side_effect=batch_search_side_effect)

    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    collections = [
        "Entity_name",
        "EdgeType_relationship_name",
        "MissingCollection",
        "EmptyCollection",
    ]

    await vector_search.embed_and_retrieve_distances(
        query=None, query_batch=query_batch, collections=collections, wide_search_limit=None
    )

    assert vector_search.query_list_length == 2
    assert len(vector_search.edge_distances) == 2
    assert vector_search.edge_distances[0] == edge_results_query_a
    assert vector_search.edge_distances[1] == edge_results_query_b
    assert len(vector_search.node_distances["Entity_name"]) == 2
    assert vector_search.node_distances["Entity_name"][0] == node_results_query_a
    assert vector_search.node_distances["Entity_name"][1] == node_results_query_b
    assert len(vector_search.node_distances["MissingCollection"]) == 2
    assert vector_search.node_distances["MissingCollection"] == [[], []]
    assert len(vector_search.node_distances["EmptyCollection"]) == 2
    assert vector_search.node_distances["EmptyCollection"] == [[], []]
    mock_vector_engine.embedding_engine.embed_text.assert_not_called()


@pytest.mark.asyncio
async def test_node_edge_vector_search_input_validation_both_provided():
    """Test that providing both query and query_batch raises ValueError."""
    mock_vector_engine = AsyncMock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    collections = ["Entity_name"]

    with pytest.raises(ValueError, match="Cannot provide both 'query' and 'query_batch'"):
        await vector_search.embed_and_retrieve_distances(
            query="test", query_batch=["test1", "test2"], collections=collections
        )


@pytest.mark.asyncio
async def test_node_edge_vector_search_input_validation_neither_provided():
    """Test that providing neither query nor query_batch raises ValueError."""
    mock_vector_engine = AsyncMock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    collections = ["Entity_name"]

    with pytest.raises(ValueError, match="Must provide either 'query' or 'query_batch'"):
        await vector_search.embed_and_retrieve_distances(
            query=None, query_batch=None, collections=collections
        )


@pytest.mark.asyncio
async def test_node_edge_vector_search_extract_relevant_node_ids_single_query():
    """Test that extract_relevant_node_ids returns IDs for single query mode."""
    mock_vector_engine = AsyncMock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    vector_search.query_list_length = None
    vector_search.node_distances = {
        "Entity_name": [MockScoredResult("node1", 0.95), MockScoredResult("node2", 0.87)],
        "TextSummary_text": [MockScoredResult("node1", 0.90), MockScoredResult("node3", 0.92)],
    }

    node_ids = vector_search.extract_relevant_node_ids()
    assert set(node_ids) == {"node1", "node2", "node3"}


@pytest.mark.asyncio
async def test_node_edge_vector_search_extract_relevant_node_ids_batch():
    """Test that extract_relevant_node_ids returns empty list for batch mode."""
    mock_vector_engine = AsyncMock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    vector_search.query_list_length = 2
    vector_search.node_distances = {
        "Entity_name": [
            [MockScoredResult("node1", 0.95)],
            [MockScoredResult("node2", 0.87)],
        ],
    }

    node_ids = vector_search.extract_relevant_node_ids()
    assert node_ids == []


@pytest.mark.asyncio
async def test_node_edge_vector_search_has_results_single_query():
    """Test has_results returns True when results exist and False when only empties."""
    mock_vector_engine = AsyncMock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)

    vector_search.edge_distances = [MockScoredResult("edge1", 0.92)]
    vector_search.node_distances = {}
    assert vector_search.has_results() is True

    vector_search.edge_distances = []
    vector_search.node_distances = {"Entity_name": [MockScoredResult("node1", 0.95)]}
    assert vector_search.has_results() is True

    vector_search.edge_distances = []
    vector_search.node_distances = {}
    assert vector_search.has_results() is False


@pytest.mark.asyncio
async def test_node_edge_vector_search_has_results_batch():
    """Test has_results works correctly for batch mode with list-of-lists."""
    mock_vector_engine = AsyncMock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    vector_search.query_list_length = 2

    vector_search.edge_distances = [[MockScoredResult("edge1", 0.92)], []]
    vector_search.node_distances = {}
    assert vector_search.has_results() is True

    vector_search.edge_distances = [[], []]
    vector_search.node_distances = {
        "Entity_name": [[MockScoredResult("node1", 0.95)], []],
    }
    assert vector_search.has_results() is True

    vector_search.edge_distances = [[], []]
    vector_search.node_distances = {"Entity_name": [[], []]}
    assert vector_search.has_results() is False


@pytest.mark.asyncio
async def test_node_edge_vector_search_single_query_collection_not_found():
    """Test that CollectionNotFoundError in single query mode returns empty list."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(
        side_effect=CollectionNotFoundError("Collection not found")
    )

    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    collections = ["MissingCollection"]

    await vector_search.embed_and_retrieve_distances(
        query="test query", query_batch=None, collections=collections, wide_search_limit=10
    )

    assert vector_search.node_distances["MissingCollection"] == []


@pytest.mark.asyncio
async def test_node_edge_vector_search_missing_collections_single_query():
    """Test that missing collections in single-query mode are handled gracefully with empty lists."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    node_result = MockScoredResult("node1", 0.95)

    def search_side_effect(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "Entity_name":
            return [node_result]
        elif collection_name == "MissingCollection":
            raise CollectionNotFoundError("Collection not found")
        return []

    mock_vector_engine.search = AsyncMock(side_effect=search_side_effect)

    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    collections = ["Entity_name", "MissingCollection", "EmptyCollection"]

    await vector_search.embed_and_retrieve_distances(
        query="test query", query_batch=None, collections=collections, wide_search_limit=10
    )

    assert len(vector_search.node_distances["Entity_name"]) == 1
    assert vector_search.node_distances["Entity_name"][0].id == "node1"
    assert vector_search.node_distances["Entity_name"][0].score == 0.95
    assert vector_search.node_distances["MissingCollection"] == []
    assert vector_search.node_distances["EmptyCollection"] == []


@pytest.mark.asyncio
async def test_node_edge_vector_search_has_results_batch_nodes_only():
    """Test has_results returns True when only node distances are populated in batch mode."""
    mock_vector_engine = AsyncMock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    vector_search.query_list_length = 2
    vector_search.edge_distances = [[], []]
    vector_search.node_distances = {
        "Entity_name": [[MockScoredResult("node1", 0.95)], []],
    }

    assert vector_search.has_results() is True


@pytest.mark.asyncio
async def test_node_edge_vector_search_has_results_batch_edges_only():
    """Test has_results returns True when only edge distances are populated in batch mode."""
    mock_vector_engine = AsyncMock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    vector_search.query_list_length = 2
    vector_search.edge_distances = [[MockScoredResult("edge1", 0.92)], []]
    vector_search.node_distances = {}

    assert vector_search.has_results() is True
