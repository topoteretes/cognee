import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch


def vector_engine_mock() -> AsyncMock:
    """A vector engine mock with the interface defaults a bare mock lacks: every
    collection exists, and a batch is scored one ``score_by_ids`` call per vector."""
    engine = AsyncMock()
    engine.existing_collections = AsyncMock(side_effect=lambda names: list(names))
    engine.score_by_ids_batch = MethodType(VectorDBInterface.score_by_ids_batch, engine)
    return engine


class MockScoredResult:
    """Mock class for vector search results."""

    def __init__(self, id, score, payload=None):
        self.id = id
        self.score = score
        self.payload = payload or {}


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_id_scoring_reuses_embeddings_and_preserves_query_shapes(batch):
    engine = vector_engine_mock()
    vectors = [[1.0, 0.0], [0.0, 1.0]] if batch else [[1.0, 0.0]]
    engine.embedding_engine.embed_text.return_value = vectors

    async def score(collection_name, data_point_ids, query_vector):
        if collection_name == "Missing_text":
            raise CollectionNotFoundError("missing")
        return [
            MockScoredResult(point_id, query_vector[0] + index)
            for index, point_id in enumerate(data_point_ids)
        ]

    engine.score_by_ids.side_effect = score
    search = NodeEdgeVectorSearch(vector_engine=engine)
    queries = ["first", "second"] if batch else ["first"]
    await search.embed_and_retrieve_distances(
        query=None if batch else queries[0],
        query_batch=queries if batch else None,
        collections=["Entity_name", search.edge_collection, "Missing_text", "Unrequested_text"],
        ids_by_collection={
            "Entity_name": ["far", "near"],
            search.edge_collection: ["edge-type"],
            "Missing_text": ["missing"],
        },
    )
    engine.embedding_engine.embed_text.assert_awaited_once_with(queries)
    engine.search.assert_not_awaited()
    engine.batch_search.assert_not_awaited()
    for call in engine.score_by_ids.call_args_list:
        assert call.kwargs["collection_name"] != "Unrequested_text"
        assert (
            call.kwargs["data_point_ids"]
            == {
                "Entity_name": ["far", "near"],
                search.edge_collection: ["edge-type"],
                "Missing_text": ["missing"],
            }[call.kwargs["collection_name"]]
        )
    nodes = search.node_distances["Entity_name"]
    edges = search.edge_distances
    if batch:
        assert search.query_list_length == 2
        assert [[row.id for row in rows] for rows in nodes] == [["far", "near"]] * 2
        assert [[row.score for row in rows] for rows in nodes] == [[1.0, 2.0], [0.0, 1.0]]
        assert [[row.id for row in rows] for rows in edges] == [["edge-type"]] * 2
        assert search.node_distances["Missing_text"] == [[], []]
        assert search.node_distances["Unrequested_text"] == [[], []]
    else:
        assert [row.id for row in nodes] == ["far", "near"]
        assert [row.score for row in nodes] == [1.0, 2.0]
        assert [row.id for row in edges] == ["edge-type"]
        assert search.node_distances["Missing_text"] == []
        assert search.node_distances["Unrequested_text"] == []


@pytest.mark.asyncio
async def test_batch_id_scoring_is_one_adapter_call_per_collection():
    """A batch hands all its query vectors to ``score_by_ids_batch`` once per collection,
    and each vector's results land in its own slot, in order."""
    engine = AsyncMock()
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    engine.embedding_engine.embed_text.return_value = vectors

    async def score_batch(collection_name, data_point_ids, query_vectors):
        if collection_name == "Missing_text":
            raise CollectionNotFoundError("missing")
        return [
            [MockScoredResult(point_id, query_vector[0]) for point_id in data_point_ids]
            for query_vector in query_vectors
        ]

    engine.score_by_ids_batch.side_effect = score_batch
    search = NodeEdgeVectorSearch(vector_engine=engine)
    await search.embed_and_retrieve_distances(
        query=None,
        query_batch=["first", "second"],
        collections=["Entity_name", "Missing_text", "Unrequested_text"],
        ids_by_collection={"Entity_name": ["a", "b"], "Missing_text": ["m"]},
    )

    engine.score_by_ids.assert_not_awaited()
    assert engine.score_by_ids_batch.await_count == 2  # Unrequested_text has no ids
    for call in engine.score_by_ids_batch.call_args_list:
        assert call.kwargs["query_vectors"] == vectors
    assert [[row.score for row in rows] for rows in search.node_distances["Entity_name"]] == [
        [1.0, 1.0],
        [0.0, 0.0],
    ]
    assert search.node_distances["Missing_text"] == [[], []]
    assert search.node_distances["Unrequested_text"] == [[], []]


@pytest.mark.asyncio
async def test_default_batch_id_scoring_runs_the_queries_of_a_batch_concurrently():
    """The interface default behind ``score_by_ids_batch`` scores a batch's query
    vectors at once, like ``batch_search``, not one after another: on LanceDB each
    ``score_by_ids`` is a scan, and serial scans multiplied the batch's cost by its size."""
    engine = vector_engine_mock()
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    engine.embedding_engine.embed_text.return_value = vectors
    in_flight = 0
    most_in_flight = 0

    async def score(collection_name, data_point_ids, query_vector):
        nonlocal in_flight, most_in_flight
        in_flight += 1
        most_in_flight = max(most_in_flight, in_flight)
        await asyncio.sleep(0)  # let the other queries of the batch start
        in_flight -= 1
        return [MockScoredResult(point_id, query_vector[0]) for point_id in data_point_ids]

    engine.score_by_ids.side_effect = score
    search = NodeEdgeVectorSearch(vector_engine=engine)
    await search.embed_and_retrieve_distances(
        query=None,
        query_batch=["first", "second", "third"],
        collections=["Entity_name"],
        ids_by_collection={"Entity_name": ["a", "b"]},
    )

    assert most_in_flight == len(vectors)
    assert [[row.score for row in rows] for rows in search.node_distances["Entity_name"]] == [
        [1.0, 1.0],
        [0.0, 0.0],
        [0.5, 0.5],
    ]


@pytest.mark.asyncio
async def test_optional_id_scoring_is_detected_before_search():
    inherited = MethodType(VectorDBInterface.score_by_ids, object())
    for engine in [SimpleNamespace(), SimpleNamespace(score_by_ids=inherited)]:
        assert not await NodeEdgeVectorSearch(vector_engine=engine).supports_id_scoring()
    assert await NodeEdgeVectorSearch(vector_engine=AsyncMock()).supports_id_scoring()


@pytest.mark.asyncio
async def test_optional_id_scoring_detection_survives_engine_leases():
    from cognee.infrastructure.databases.utils.closing_lru_cache import closing_lru_cache

    class Adapter:
        score_by_ids = VectorDBInterface.score_by_ids

    @closing_lru_cache(maxsize=1)
    def cached():
        return Adapter()

    try:
        assert not await NodeEdgeVectorSearch(vector_engine=cached()).supports_id_scoring()
    finally:
        cached.cache_clear()


@pytest.mark.asyncio
async def test_id_scoring_failure_does_not_retry_as_a_global_search():
    engine = vector_engine_mock()
    engine.embedding_engine.embed_text.return_value = [[1.0, 0.0]]
    engine.score_by_ids.side_effect = RuntimeError("store unavailable")
    search = NodeEdgeVectorSearch(vector_engine=engine)
    with pytest.raises(RuntimeError, match="store unavailable"):
        await search.embed_and_retrieve_distances(
            query="test", collections=["Entity_name"], ids_by_collection={"Entity_name": ["id"]}
        )
    engine.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_node_edge_vector_search_single_query_shape():
    """Test that single query mode produces flat lists (not list-of-lists)."""
    mock_vector_engine = vector_engine_mock()
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
    mock_vector_engine = vector_engine_mock()
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
    mock_vector_engine = vector_engine_mock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    collections = ["Entity_name"]

    with pytest.raises(ValueError, match="Cannot provide both 'query' and 'query_batch'"):
        await vector_search.embed_and_retrieve_distances(
            query="test", query_batch=["test1", "test2"], collections=collections
        )


@pytest.mark.asyncio
async def test_node_edge_vector_search_input_validation_neither_provided():
    """Test that providing neither query nor query_batch raises ValueError."""
    mock_vector_engine = vector_engine_mock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    collections = ["Entity_name"]

    with pytest.raises(ValueError, match="Must provide either 'query' or 'query_batch'"):
        await vector_search.embed_and_retrieve_distances(
            query=None, query_batch=None, collections=collections
        )


@pytest.mark.asyncio
async def test_node_edge_vector_search_extract_relevant_node_ids_single_query():
    """Test that extract_relevant_node_ids returns IDs for single query mode."""
    mock_vector_engine = vector_engine_mock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    vector_search.query_list_length = None
    vector_search.node_distances = {
        "Entity_name": [MockScoredResult("node1", 0.95), MockScoredResult("node2", 0.87)],
        "TextSummary_text": [MockScoredResult("node1", 0.90), MockScoredResult("node3", 0.92)],
    }

    node_ids = vector_search.extract_relevant_node_ids()
    assert set(node_ids) == {"node1", "node2", "node3"}


def test_node_edge_vector_search_extract_relevant_node_ids_ordered_by_score():
    """Node IDs come back nearest first (lowest distance), using each node's best score."""
    vector_search = NodeEdgeVectorSearch(vector_engine=AsyncMock())
    vector_search.query_list_length = None
    vector_search.node_distances = {
        "Entity_name": [
            MockScoredResult("far", 0.9),
            MockScoredResult("near", 0.1),
            MockScoredResult("shared", 0.8),
        ],
        "TextSummary_text": [MockScoredResult("shared", 0.2), MockScoredResult("mid", 0.5)],
    }

    assert vector_search.extract_relevant_node_ids() == ["near", "shared", "mid", "far"]


@pytest.mark.asyncio
async def test_node_edge_vector_search_extract_relevant_node_ids_batch():
    """Test that extract_relevant_node_ids returns empty list for batch mode."""
    mock_vector_engine = vector_engine_mock()
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
    mock_vector_engine = vector_engine_mock()
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
    mock_vector_engine = vector_engine_mock()
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
    mock_vector_engine = vector_engine_mock()
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
    mock_vector_engine = vector_engine_mock()
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
    mock_vector_engine = vector_engine_mock()
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
    mock_vector_engine = vector_engine_mock()
    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    vector_search.query_list_length = 2
    vector_search.edge_distances = [[MockScoredResult("edge1", 0.92)], []]
    vector_search.node_distances = {}

    assert vector_search.has_results() is True


@pytest.mark.asyncio
async def test_node_set_filter_skips_edge_type_collection():
    """Node collections keep the node-set filter; edge types never get it, since they only
    score edges the graph projection already scoped (gh #4822)."""
    mock_vector_engine = vector_engine_mock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_vector_engine.search = AsyncMock(return_value=[MockScoredResult("hit", 0.5)])

    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    await vector_search.embed_and_retrieve_distances(
        query="test query",
        collections=["Entity_name", "EdgeType_relationship_name"],
        node_name=["A"],
        node_name_filter_operator="AND",
    )

    filters = {
        call.kwargs["collection_name"]: (
            call.kwargs["node_name"],
            call.kwargs["node_name_filter_operator"],
        )
        for call in mock_vector_engine.search.await_args_list
    }
    assert filters == {
        "Entity_name": (["A"], "AND"),
        "EdgeType_relationship_name": (None, "AND"),
    }
    assert vector_search.edge_distances


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_missing_collections_are_not_searched(batch):
    """Collections the store lacks answer empty without a search call, in input order."""
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2]])
    mock_vector_engine.existing_collections = AsyncMock(return_value=["Entity_name"])
    hit = MockScoredResult("node-1", 0.1)
    mock_vector_engine.search = AsyncMock(return_value=[hit])
    mock_vector_engine.batch_search = AsyncMock(return_value=[[hit]])

    vector_search = NodeEdgeVectorSearch(vector_engine=mock_vector_engine)
    collections = ["Missing_text", "Entity_name", "EdgeType_relationship_name"]
    if batch:
        await vector_search.embed_and_retrieve_distances(query_batch=["q"], collections=collections)
        searched = [
            c.kwargs["collection_name"] for c in mock_vector_engine.batch_search.await_args_list
        ]
        assert vector_search.node_distances == {"Missing_text": [[]], "Entity_name": [[hit]]}
        assert vector_search.edge_distances == [[]]
    else:
        await vector_search.embed_and_retrieve_distances(query="q", collections=collections)
        searched = [c.kwargs["collection_name"] for c in mock_vector_engine.search.await_args_list]
        assert vector_search.node_distances == {"Missing_text": [], "Entity_name": [hit]}
        assert vector_search.edge_distances == []

    mock_vector_engine.existing_collections.assert_awaited_once_with(collections)
    assert searched == ["Entity_name"]
