"""Unit tests for NeptuneAnalyticsAdapter.add_nodes_with_vectors and add_edges_with_vectors.

These tests verify the two new hybrid-write methods. They are skipped automatically
when the Neptune optional dependencies (langchain_aws, botocore) are not installed.
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
