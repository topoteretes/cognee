"""Unit tests for ChromaDBAdapter search filtering."""

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

chromadb = pytest.importorskip("chromadb", reason="ChromaDB tests require cognee[chromadb]")

from cognee.infrastructure.databases.vector.adapters.chromadb.ChromaDBAdapter import (  # noqa: E402
    BELONGS_TO_SET_KEY,
    ChromaDBAdapter,
)


def _make_chroma_result(ids, distances, metadatas=None):
    result = {"ids": [ids], "distances": [distances]}
    if metadatas is not None:
        result["metadatas"] = [metadatas]
    return result


def _make_adapter():
    adapter = ChromaDBAdapter.__new__(ChromaDBAdapter)
    adapter.embedding_engine = AsyncMock()
    adapter.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_collection = AsyncMock()
    mock_collection.count = AsyncMock(return_value=10)
    adapter.get_collection = AsyncMock(return_value=mock_collection)
    adapter.embed_data = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return adapter, mock_collection


@pytest.mark.asyncio
async def test_search_node_name_applies_contains_filter():
    adapter, collection = _make_adapter()
    collection.query = AsyncMock(return_value=_make_chroma_result(ids=[], distances=[]))

    await adapter.search(
        collection_name="test_col",
        query_text="hello",
        limit=5,
        include_payload=False,
        node_name=["Alice"],
    )

    where = collection.query.call_args[1]["where"]
    assert where == {BELONGS_TO_SET_KEY: {"$contains": "Alice"}}


@pytest.mark.asyncio
async def test_search_include_payload_true_fetches_metadatas():
    adapter, collection = _make_adapter()
    node_id = str(uuid4())
    collection.query = AsyncMock(
        return_value=_make_chroma_result(
            ids=[node_id],
            distances=[0.2],
            metadatas=[{"text": "Alice", "id": node_id, "belongs_to_set": ["Alice"]}],
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
    assert "metadatas" in collection.query.call_args[1]["include"]
