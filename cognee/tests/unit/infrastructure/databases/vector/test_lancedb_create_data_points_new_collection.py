"""Regression: create_data_points must not deadlock when the collection is new.

``create_data_points`` used to acquire ``VECTOR_DB_LOCK`` and then call
``create_collection``, which tried to acquire the same non-reentrant
``asyncio.Lock`` again — hanging forever (#5987).
"""

from __future__ import annotations

import asyncio

import pytest

from cognee.infrastructure.engine import DataPoint

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        LanceDBAdapter,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _Note(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_create_data_points_on_new_collection_does_not_deadlock(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    collection = "notes"
    point = _Note(text="x")

    assert not await adapter.has_collection(collection)

    # Before the fix this never returned (nested VECTOR_DB_LOCK acquire).
    await asyncio.wait_for(
        adapter.create_data_points(collection, [point]),
        timeout=20,
    )

    assert await adapter.has_collection(collection)
    results = await adapter.retrieve(collection, [str(point.id)])
    assert len(results) == 1
    assert results[0].payload["text"] == "x"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_create_data_points_after_create_collection_still_works(tmp_path):
    """Sanity: pre-creating the collection (old happy path) still upserts."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    collection = "notes"
    point = _Note(text="y")

    await adapter.create_collection(collection, type(point))
    await asyncio.wait_for(
        adapter.create_data_points(collection, [point]),
        timeout=20,
    )

    results = await adapter.retrieve(collection, [str(point.id)])
    assert len(results) == 1
    assert results[0].payload["text"] == "y"
