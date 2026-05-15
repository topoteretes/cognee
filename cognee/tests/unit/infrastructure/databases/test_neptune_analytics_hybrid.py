"""Unit tests for NeptuneAnalyticsAdapter hybrid-write methods and search().

- add_nodes_with_vectors / add_edges_with_vectors
- search() include_payload handling and node_name filtering

These tests verify the hybrid-write methods and search behaviour. They are skipped
automatically when the Neptune optional dependencies (langchain_aws, botocore) are
not installed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

langchain_aws = pytest.importorskip(
    "langchain_aws", reason="Neptune Analytics tests require langchain_aws"
)
botocore = pytest.importorskip("botocore", reason="Neptune Analytics tests require botocore")

from cognee.infrastructure.engine import DataPoint  # noqa: E402
from cognee.infrastructure.databases.hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (  # noqa: E402
    NeptuneAnalyticsAdapter,
)


class SimpleNode(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class NodeNoIndex(DataPoint):
    name: str
    metadata: dict = {"index_fields": []}


class _FakeAdapter:
    """Minimal stand-in for NeptuneAnalyticsAdapter that carries the two new methods."""

    add_nodes_with_vectors = NeptuneAnalyticsAdapter.add_nodes_with_vectors
    add_edges_with_vectors = NeptuneAnalyticsAdapter.add_edges_with_vectors

    def __init__(self):
        self.add_nodes = AsyncMock()
        self.add_edges = AsyncMock()
        self.create_vector_index = AsyncMock()
        self.create_data_points = AsyncMock()


# ---------------------------------------------------------------------------
# add_nodes_with_vectors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_nodes_with_vectors_empty():
    """Empty list must be a no-op — add_nodes and create_data_points not called."""
    adapter = _FakeAdapter()

    await adapter.add_nodes_with_vectors([])

    adapter.add_nodes.assert_not_awaited()
    adapter.create_data_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_nodes_with_vectors_calls_add_nodes_and_create_data_points():
    """Nodes are inserted into graph and one vector collection created per index field."""
    adapter = _FakeAdapter()
    node = SimpleNode(id=uuid4(), name="Alice")

    await adapter.add_nodes_with_vectors([node])

    adapter.add_nodes.assert_awaited_once_with([node])
    adapter.create_vector_index.assert_awaited_once_with("SimpleNode", "name")
    adapter.create_data_points.assert_awaited_once()

    collection_arg = adapter.create_data_points.call_args[0][0]
    assert collection_arg == "SimpleNode_name"

    schemas = adapter.create_data_points.call_args[0][1]
    assert len(schemas) == 1
    assert str(schemas[0].id) == str(node.id)
    assert schemas[0].text == "Alice"


@pytest.mark.asyncio
async def test_add_nodes_with_vectors_skips_nodes_without_index_fields():
    """Nodes with no index_fields must not trigger create_data_points."""
    adapter = _FakeAdapter()
    node = NodeNoIndex(id=uuid4(), name="Bob")

    await adapter.add_nodes_with_vectors([node])

    adapter.add_nodes.assert_awaited_once_with([node])
    adapter.create_data_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_nodes_with_vectors_multiple_nodes_same_type():
    """Multiple nodes of the same type produce one collection call with all schemas."""
    adapter = _FakeAdapter()
    node_a = SimpleNode(id=uuid4(), name="Alice")
    node_b = SimpleNode(id=uuid4(), name="Bob")

    await adapter.add_nodes_with_vectors([node_a, node_b])

    adapter.add_nodes.assert_awaited_once_with([node_a, node_b])
    adapter.create_vector_index.assert_awaited_once_with("SimpleNode", "name")
    adapter.create_data_points.assert_awaited_once()

    schemas = adapter.create_data_points.call_args[0][1]
    assert len(schemas) == 2
    texts = {s.text for s in schemas}
    assert texts == {"Alice", "Bob"}


# ---------------------------------------------------------------------------
# add_edges_with_vectors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_edges_with_vectors_empty():
    """Empty edge list must be a no-op."""
    adapter = _FakeAdapter()

    await adapter.add_edges_with_vectors([])

    adapter.add_edges.assert_not_awaited()
    adapter.create_data_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_edges_with_vectors_calls_add_edges_and_creates_edge_type_vectors():
    """Edges are inserted into graph and one EdgeType schema per unique relationship."""
    adapter = _FakeAdapter()
    src, tgt = str(uuid4()), str(uuid4())
    edges = [
        (src, tgt, "knows", {"source_node_id": src, "target_node_id": tgt}),
        (src, tgt, "knows", {"source_node_id": src, "target_node_id": tgt}),
        (tgt, src, "likes", {"source_node_id": tgt, "target_node_id": src}),
    ]

    await adapter.add_edges_with_vectors(edges)

    adapter.add_edges.assert_awaited_once_with(edges)
    adapter.create_vector_index.assert_awaited_once_with("EdgeType", "relationship_name")
    adapter.create_data_points.assert_awaited_once()

    collection_arg = adapter.create_data_points.call_args[0][0]
    assert collection_arg == "EdgeType_relationship_name"

    schemas = adapter.create_data_points.call_args[0][1]
    assert len(schemas) == 2
    texts = {s.text for s in schemas}
    assert texts == {"knows", "likes"}


@pytest.mark.asyncio
async def test_add_edges_with_vectors_uses_edge_text_property_when_present():
    """edge_text in edge properties overrides relationship_name for embedding text."""
    adapter = _FakeAdapter()
    src, tgt = str(uuid4()), str(uuid4())
    edges = [(src, tgt, "rel", {"edge_text": "custom text"})]

    await adapter.add_edges_with_vectors(edges)

    schemas = adapter.create_data_points.call_args[0][1]
    assert len(schemas) == 1
    assert schemas[0].text == "custom text"


# ---------------------------------------------------------------------------
# search: include_payload and node_name filtering
# ---------------------------------------------------------------------------


def _re_raise_exception(e, _query):
    raise e


class _FakeSearchAdapter:
    """Minimal stand-in for NeptuneAnalyticsAdapter.search tests."""

    search = NeptuneAnalyticsAdapter.search

    def __init__(self, query_response=None):
        self._client = MagicMock()
        self._client.query = MagicMock(return_value=query_response or [])
        self.embedding_engine = AsyncMock()
        self.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        self._COLLECTION_PREFIX = "collection"
        self._TOPK_LOWER_BOUND = 0
        self._TOPK_UPPER_BOUND = 100
        self._na_exception_handler = MagicMock(side_effect=_re_raise_exception)
        self._validate_embedding_engine = MagicMock()


@pytest.mark.asyncio
async def test_search_include_payload_false_omits_node_properties():
    """When include_payload=False, query uses id(node) and payload is None."""
    node_id = str(uuid4())
    fake_response = [{"node_id": node_id, "score": 0.9}]
    adapter = _FakeSearchAdapter(query_response=fake_response)

    results = await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=False,
    )

    assert len(results) == 1
    assert results[0].payload is None
    assert results[0].score == 0.9

    # Verify the query sent to Neptune does NOT fetch full node properties
    query_sent = adapter._client.query.call_args[0][0]
    assert "id(node) as node_id" in query_sent
    assert "node as payload" not in query_sent


@pytest.mark.asyncio
async def test_search_include_payload_true_returns_node_properties():
    """When include_payload=True, query uses node as payload and payload is populated."""
    node_id = str(uuid4())
    fake_response = [{"payload": {"~id": node_id, "~properties": {"name": "Alice"}}, "score": 0.8}]
    adapter = _FakeSearchAdapter(query_response=fake_response)

    results = await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=True,
    )

    assert len(results) == 1
    assert results[0].payload == {"name": "Alice"}
    assert results[0].score == 0.8

    query_sent = adapter._client.query.call_args[0][0]
    assert "node as payload" in query_sent
    assert "id(node) as node_id" not in query_sent


@pytest.mark.asyncio
async def test_search_node_name_filter_or_adds_where_clause():
    """node_name with OR operator adds an 'any(...) WHERE' filter to the query."""
    adapter = _FakeSearchAdapter(query_response=[])

    await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=False,
        node_name=["Alice", "Bob"],
        node_name_filter_operator="OR",
    )

    query_sent = adapter._client.query.call_args[0][0]
    assert "any(name IN node.belongs_to_set" in query_sent
    assert "'Alice'" in query_sent
    assert "'Bob'" in query_sent


@pytest.mark.asyncio
async def test_search_node_name_filter_and_adds_all_clause():
    """node_name with AND operator adds an 'all(...) WHERE' filter to the query."""
    adapter = _FakeSearchAdapter(query_response=[])

    await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=False,
        node_name=["Alice", "Bob"],
        node_name_filter_operator="AND",
    )

    query_sent = adapter._client.query.call_args[0][0]
    assert "all(name IN node.belongs_to_set" in query_sent


# ---------------------------------------------------------------------------
# search: with_vector=True branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_with_vector_true_include_payload_false_contains_embedding():
    """with_vector=True, include_payload=False: query has embedding in RETURN and vector in result."""
    node_id = str(uuid4())
    fake_embedding = [0.5, 0.6, 0.7]
    fake_response = [{"node_id": node_id, "score": 0.7, "embedding": fake_embedding}]
    adapter = _FakeSearchAdapter(query_response=fake_response)

    results = await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=False,
        with_vector=True,
    )

    assert len(results) == 1
    assert results[0].payload is None

    query_sent = adapter._client.query.call_args[0][0]
    assert ", embedding" in query_sent.split("RETURN", 1)[1]
    assert "neptune.algo.vectors.get" in query_sent
    assert "id(node) as node_id" in query_sent
    assert "node as payload" not in query_sent


@pytest.mark.asyncio
async def test_search_with_vector_true_include_payload_true_contains_embedding_and_payload():
    """with_vector=True, include_payload=True: query returns full node and embedding."""
    node_id = str(uuid4())
    fake_embedding = [0.5, 0.6, 0.7]
    fake_response = [
        {
            "payload": {"~id": node_id, "~properties": {"name": "Bob"}},
            "score": 0.6,
            "embedding": fake_embedding,
        }
    ]
    adapter = _FakeSearchAdapter(query_response=fake_response)

    results = await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=True,
        with_vector=True,
    )

    assert len(results) == 1
    assert results[0].payload == {"name": "Bob"}

    query_sent = adapter._client.query.call_args[0][0]
    assert "embedding" in query_sent
    assert "neptune.algo.vectors.get" in query_sent
    assert "node as payload" in query_sent
    assert "id(node) as node_id" not in query_sent
