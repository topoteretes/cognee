"""Integration tests for PGVector belongs_to_set upsert and detag.

Exercises the wire-level behavior of `PGVectorAdapter.create_data_points`
and `remove_belongs_to_set_tags` against a live Postgres+pgvector instance,
without touching LLMs or embeddings — `embed_data` is stubbed to return a
fixed vector so the tests cost nothing to run.

Skipped unless `TEST_PGVECTOR_URL` is set, e.g.:

    TEST_PGVECTOR_URL="postgresql+asyncpg://cognee:cognee@127.0.0.1:5432/cognee_db"
"""

from __future__ import annotations

import os
from typing import List
from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint

PG_URL = os.getenv("TEST_PGVECTOR_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL,
    reason="TEST_PGVECTOR_URL not set; skipping live pgvector integration tests",
)


try:
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
    )

    HAS_PGVECTOR = True
except ModuleNotFoundError:
    HAS_PGVECTOR = False


class _FakeEmbeddingEngine:
    """Deterministic no-API-call embedding engine for integration tests."""

    def get_vector_size(self) -> int:
        return 3

    def get_batch_size(self) -> int:
        return 100

    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _TaggedPoint(DataPoint):
    """Minimal DataPoint used to exercise live pgvector `belongs_to_set` semantics."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


async def _fresh_adapter() -> PGVectorAdapter:
    """Build an adapter pointed at the live pgvector instance."""
    adapter = PGVectorAdapter(
        connection_string=PG_URL,
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    return adapter


@pytest.fixture
def collection_name() -> str:
    """Return a unique PascalCase collection name per test to isolate runs."""
    return f"IntegTaggedPoint_{uuid4().hex[:10]}_text"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra not installed")
async def test_create_data_points_merges_belongs_to_set(collection_name):
    """Re-upserting the same id with a new tag must union the tags in pgvector."""
    adapter = await _fresh_adapter()
    point_id = uuid4()

    try:
        await adapter.create_data_points(
            collection_name,
            [_TaggedPoint(id=point_id, text="shared", belongs_to_set=["DatasetA"])],
        )
        await adapter.create_data_points(
            collection_name,
            [_TaggedPoint(id=point_id, text="shared", belongs_to_set=["DatasetB"])],
        )

        results = await adapter.retrieve(collection_name, [str(point_id)])
        assert len(results) == 1
        assert sorted(results[0].payload["belongs_to_set"]) == ["DatasetA", "DatasetB"]
    finally:
        await adapter.delete_data_points(collection_name, [str(point_id)])


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra not installed")
async def test_create_data_points_dedupes_duplicate_ids_in_batch(collection_name):
    """With ON CONFLICT DO UPDATE, repeating the same id in one INSERT batch
    would otherwise fail with "cannot affect row a second time". The adapter
    must dedup in Python before the insert."""
    adapter = await _fresh_adapter()
    point_id = uuid4()

    try:
        await adapter.create_data_points(
            collection_name,
            [
                _TaggedPoint(id=point_id, text="shared", belongs_to_set=["DatasetA"]),
                _TaggedPoint(id=point_id, text="shared", belongs_to_set=["DatasetA"]),
            ],
        )

        results = await adapter.retrieve(collection_name, [str(point_id)])
        assert len(results) == 1
        assert results[0].payload["belongs_to_set"] == ["DatasetA"]
    finally:
        await adapter.delete_data_points(collection_name, [str(point_id)])


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra not installed")
async def test_remove_belongs_to_set_tags_strips_and_deletes(collection_name):
    """Detag strips the target tag, removes rows that empty out, and leaves others alone."""
    adapter = await _fresh_adapter()

    shared_id = uuid4()
    orphaned_id = uuid4()
    untouched_id = uuid4()

    try:
        await adapter.create_data_points(
            collection_name,
            [
                _TaggedPoint(id=shared_id, text="shared", belongs_to_set=["Dev", "DevMirror"]),
                _TaggedPoint(id=orphaned_id, text="orphaned", belongs_to_set=["Dev"]),
                _TaggedPoint(id=untouched_id, text="untouched", belongs_to_set=["Production"]),
            ],
        )

        await adapter.remove_belongs_to_set_tags(["Dev"])

        surviving = await adapter.retrieve(collection_name, [str(shared_id), str(untouched_id)])
        by_id = {str(r.id): r for r in surviving}

        assert sorted(by_id[str(shared_id)].payload["belongs_to_set"]) == ["DevMirror"]
        assert by_id[str(untouched_id)].payload["belongs_to_set"] == ["Production"]
        assert await adapter.retrieve(collection_name, [str(orphaned_id)]) == []
    finally:
        await adapter.delete_data_points(
            collection_name, [str(shared_id), str(untouched_id), str(orphaned_id)]
        )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra not installed")
async def test_remove_belongs_to_set_tags_ignores_non_vector_tables():
    """The PascalCase filter must skip lowercase relational tables that share
    the schema with vector collections (e.g. `users`, `nodes`). This test
    just asserts the call completes without error; relational tables either
    don't exist or don't have a jsonb `payload` column."""
    adapter = await _fresh_adapter()
    await adapter.remove_belongs_to_set_tags(["NonexistentTag"])
