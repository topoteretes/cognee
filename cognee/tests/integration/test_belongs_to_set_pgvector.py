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
        """Return the fixed vector dimensionality used throughout these tests."""
        return 3

    def get_batch_size(self) -> int:
        """Return the stub embedding batch size."""
        return 100

    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Return a fixed 3-D vector per input text without making any API call."""
        return [[0.1, 0.2, 0.3] for _ in texts]


class _TaggedPoint(DataPoint):
    """Minimal DataPoint used to exercise live pgvector `belongs_to_set` semantics."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


def _fresh_adapter() -> PGVectorAdapter:
    """Build an adapter pointed at the live pgvector instance."""
    return PGVectorAdapter(
        connection_string=PG_URL,
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )


async def _drop_collection(adapter: PGVectorAdapter, name: str) -> None:
    """Drop the pgvector collection table so tests don't leak empty tables across CI runs."""
    from sqlalchemy import text as sql_text

    async with adapter.get_async_session() as session:
        await session.execute(sql_text(f'DROP TABLE IF EXISTS "{name}"'))
        await session.commit()
    adapter.reset_metadata_cache()


@pytest.fixture
def collection_name() -> str:
    """Return a unique PascalCase collection name per test to isolate runs."""
    return f"IntegTaggedPoint_{uuid4().hex[:10]}_text"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra not installed")
async def test_create_data_points_merges_belongs_to_set(collection_name):
    """Re-upserting the same id with a new tag must union the tags in pgvector."""
    adapter = _fresh_adapter()
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
        await _drop_collection(adapter, collection_name)


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra not installed")
async def test_create_data_points_dedupes_duplicate_ids_in_batch(collection_name):
    """With ON CONFLICT DO UPDATE, repeating the same id in one INSERT batch
    would otherwise fail with "cannot affect row a second time". The adapter
    must dedup in Python before the insert."""
    adapter = _fresh_adapter()
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
        await _drop_collection(adapter, collection_name)


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra not installed")
async def test_create_data_points_merges_tags_across_in_batch_duplicates(collection_name):
    """When the same id appears in one batch with different `belongs_to_set`
    values, the Python-side dedup must union the tags instead of keeping
    only the last duplicate's — mirrors the fix in Neo4jAdapter.add_nodes."""
    adapter = _fresh_adapter()
    point_id = uuid4()

    try:
        await adapter.create_data_points(
            collection_name,
            [
                _TaggedPoint(id=point_id, text="shared", belongs_to_set=["DatasetA"]),
                _TaggedPoint(id=point_id, text="shared", belongs_to_set=["DatasetB"]),
            ],
        )

        results = await adapter.retrieve(collection_name, [str(point_id)])
        assert len(results) == 1
        assert sorted(results[0].payload["belongs_to_set"]) == ["DatasetA", "DatasetB"]
    finally:
        await adapter.delete_data_points(collection_name, [str(point_id)])
        await _drop_collection(adapter, collection_name)


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra not installed")
async def test_remove_belongs_to_set_tags_strips_and_deletes(collection_name):
    """Detag strips the target tag, removes rows that empty out, and leaves others alone."""
    adapter = _fresh_adapter()

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
        await _drop_collection(adapter, collection_name)


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra not installed")
async def test_remove_belongs_to_set_tags_ignores_non_vector_tables():
    """Create a snake_case relational-style table and a PascalCase table with
    no `payload` column, then run detag. Both must survive with their rows
    and schema intact — the PascalCase filter and per-table error handling
    together should keep non-vector tables out of the detag path."""
    adapter = _fresh_adapter()

    from sqlalchemy import text as sql_text

    lowercase_name = f"integ_snake_{uuid4().hex[:10]}"
    pascal_noise_name = f"IntegNoPayload_{uuid4().hex[:10]}_text"

    try:
        async with adapter.get_async_session() as session:
            await session.execute(
                sql_text(f'CREATE TABLE "{lowercase_name}" (id INTEGER PRIMARY KEY, name TEXT)')
            )
            await session.execute(
                sql_text(f"INSERT INTO \"{lowercase_name}\" (id, name) VALUES (1, 'kept')")
            )
            await session.execute(
                sql_text(f'CREATE TABLE "{pascal_noise_name}" (id INTEGER PRIMARY KEY, note TEXT)')
            )
            await session.execute(
                sql_text(f"INSERT INTO \"{pascal_noise_name}\" (id, note) VALUES (1, 'kept')")
            )
            await session.commit()

        await adapter.remove_belongs_to_set_tags(["Dev"])

        async with adapter.get_async_session() as session:
            lower_count = (
                await session.execute(sql_text(f'SELECT COUNT(*) FROM "{lowercase_name}"'))
            ).scalar_one()
            pascal_count = (
                await session.execute(sql_text(f'SELECT COUNT(*) FROM "{pascal_noise_name}"'))
            ).scalar_one()

        assert lower_count == 1, "Snake-case relational table should be skipped entirely"
        assert pascal_count == 1, (
            "PascalCase table without a jsonb payload column must survive the per-table "
            "try/except in remove_belongs_to_set_tags"
        )
    finally:
        async with adapter.get_async_session() as session:
            await session.execute(sql_text(f'DROP TABLE IF EXISTS "{lowercase_name}"'))
            await session.execute(sql_text(f'DROP TABLE IF EXISTS "{pascal_noise_name}"'))
            await session.commit()
