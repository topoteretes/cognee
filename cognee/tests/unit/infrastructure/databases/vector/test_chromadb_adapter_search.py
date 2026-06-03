"""Unit tests for ChromaDBAdapter.search() and batch_search().

Covers the include_payload flag (omit metadatas when False) and the
node_name filtering support added in issue #2353.
"""

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

chromadb = pytest.importorskip("chromadb", reason="ChromaDB tests require chromadb")

from cognee.infrastructure.databases.vector.chromadb.ChromaDBAdapter import (  # noqa: E402
    BELONGS_TO_SET_MEMBER_PREFIX,
    ChromaDBAdapter,
    process_data_for_chroma,
    restore_data_from_chroma,
)


def _member_key(name):
    return f"{BELONGS_TO_SET_MEMBER_PREFIX}{name}"


def _make_chroma_result(ids, distances, metadatas=None, embeddings=None):
    """Build a ChromaDB-style query result dict."""
    result = {
        "ids": [ids],
        "distances": [distances],
    }
    if metadatas is not None:
        result["metadatas"] = [metadatas]
    if embeddings is not None:
        result["embeddings"] = [embeddings]
    return result


def _make_adapter():
    """Return a ChromaDBAdapter with mocked internals."""
    adapter = ChromaDBAdapter.__new__(ChromaDBAdapter)
    adapter.embedding_engine = AsyncMock()
    adapter.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    mock_collection = AsyncMock()
    mock_collection.count = AsyncMock(return_value=10)

    adapter.get_collection = AsyncMock(return_value=mock_collection)
    adapter.embed_data = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return adapter, mock_collection


@pytest.mark.asyncio
async def test_search_include_payload_false_omits_metadatas():
    """include_payload=False must not request metadatas from ChromaDB."""
    adapter, collection = _make_adapter()
    node_id = str(uuid4())

    collection.query = AsyncMock(
        return_value=_make_chroma_result(
            ids=[node_id],
            distances=[0.2],
        )
    )

    results = await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=False,
    )

    assert len(results) == 1
    assert results[0].payload is None

    call_kwargs = collection.query.call_args[1]
    assert "metadatas" not in call_kwargs["include"]


@pytest.mark.asyncio
async def test_search_include_payload_true_fetches_metadatas():
    """include_payload=True must request metadatas and populate payload."""
    adapter, collection = _make_adapter()
    node_id = str(uuid4())

    collection.query = AsyncMock(
        return_value=_make_chroma_result(
            ids=[node_id],
            distances=[0.2],
            metadatas=[{"text": "Alice", "id": node_id}],
        )
    )

    results = await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=True,
    )

    assert len(results) == 1
    assert results[0].payload is not None

    call_kwargs = collection.query.call_args[1]
    assert "metadatas" in call_kwargs["include"]


@pytest.mark.asyncio
async def test_search_node_name_single_applies_eq_filter():
    """A single node_name entry uses a simple equality where filter."""
    adapter, collection = _make_adapter()

    collection.query = AsyncMock(return_value=_make_chroma_result(ids=[], distances=[]))

    await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=False,
        node_name=["Alice"],
    )

    call_kwargs = collection.query.call_args[1]
    assert "where" in call_kwargs
    assert call_kwargs["where"] == {_member_key("Alice"): {"$eq": True}}


def test_belongs_to_set_is_flattened_into_boolean_membership_keys():
    """Regression for #2353/#2947/#2948: belongs_to_set must be stored as
    queryable per-tag boolean keys, not only a JSON-encoded list that the
    where filter can never match."""
    stored = process_data_for_chroma(
        {"id": "n1", "belongs_to_set": ["Alice", "Bob"], "text": "hello"}
    )

    # Each tag gets a boolean membership marker that an $eq True filter targets.
    assert stored[_member_key("Alice")] is True
    assert stored[_member_key("Bob")] is True
    # The original list is preserved so payloads round-trip unchanged.
    assert "belongs_to_set__list" in stored


def test_where_filter_targets_stored_membership_key():
    """The where filter key produced for a node_name must be a key that
    process_data_for_chroma actually writes, otherwise the filter is silently
    dropped (the original #2353 bug)."""
    stored = process_data_for_chroma({"belongs_to_set": ["Alice"]})
    where = ChromaDBAdapter._build_where_filter(["Alice"], "OR")

    (filter_key,) = where.keys()
    assert filter_key in stored
    assert where[filter_key] == {"$eq": True}


def test_belongs_to_set_round_trips_without_membership_markers():
    """restore_data_from_chroma must drop the internal membership markers and
    rebuild belongs_to_set from its JSON list."""
    stored = process_data_for_chroma({"id": "n1", "belongs_to_set": ["Alice", "Bob"]})
    restored = restore_data_from_chroma(stored)

    assert restored["belongs_to_set"] == ["Alice", "Bob"]
    assert not any(k.startswith(BELONGS_TO_SET_MEMBER_PREFIX) for k in restored)


@pytest.mark.asyncio
async def test_search_node_name_or_operator_uses_dollar_or():
    """Multiple node_names with OR operator uses $or filter."""
    adapter, collection = _make_adapter()

    collection.query = AsyncMock(return_value=_make_chroma_result(ids=[], distances=[]))

    await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=False,
        node_name=["Alice", "Bob"],
        node_name_filter_operator="OR",
    )

    call_kwargs = collection.query.call_args[1]
    assert "where" in call_kwargs
    where = call_kwargs["where"]
    assert "$or" in where
    assert where["$or"] == [
        {_member_key("Alice"): {"$eq": True}},
        {_member_key("Bob"): {"$eq": True}},
    ]


@pytest.mark.asyncio
async def test_search_node_name_and_operator_uses_dollar_and():
    """Multiple node_names with AND operator uses $and filter."""
    adapter, collection = _make_adapter()

    collection.query = AsyncMock(return_value=_make_chroma_result(ids=[], distances=[]))

    await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=False,
        node_name=["Alice", "Bob"],
        node_name_filter_operator="AND",
    )

    call_kwargs = collection.query.call_args[1]
    assert "where" in call_kwargs
    where = call_kwargs["where"]
    assert "$and" in where
    assert where["$and"] == [
        {_member_key("Alice"): {"$eq": True}},
        {_member_key("Bob"): {"$eq": True}},
    ]


@pytest.mark.asyncio
async def test_search_no_node_name_omits_where():
    """When node_name is None, no where clause is added to the query."""
    adapter, collection = _make_adapter()

    collection.query = AsyncMock(return_value=_make_chroma_result(ids=[], distances=[]))

    await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=False,
    )

    call_kwargs = collection.query.call_args[1]
    assert "where" not in call_kwargs


@pytest.mark.asyncio
async def test_batch_search_include_payload_false_omits_metadatas():
    """batch_search with include_payload=False must not request metadatas."""
    adapter, collection = _make_adapter()
    node_id = str(uuid4())

    collection.query = AsyncMock(
        return_value={
            "ids": [[node_id]],
            "distances": [[0.3]],
        }
    )

    results = await adapter.batch_search(
        collection_name="test_col",
        query_texts=["hello"],
        limit=5,
        include_payload=False,
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].payload is None

    call_kwargs = collection.query.call_args[1]
    assert "metadatas" not in call_kwargs["include"]


@pytest.mark.asyncio
async def test_batch_search_node_name_filter_is_forwarded():
    """batch_search forwards node_name filter to the ChromaDB query."""
    adapter, collection = _make_adapter()

    collection.query = AsyncMock(return_value={"ids": [[]], "distances": [[]]})

    await adapter.batch_search(
        collection_name="test_col",
        query_texts=["hello"],
        limit=5,
        include_payload=False,
        node_name=["Alice"],
    )

    call_kwargs = collection.query.call_args[1]
    assert "where" in call_kwargs
    assert call_kwargs["where"] == {_member_key("Alice"): {"$eq": True}}
