"""Concurrent pgvector inserts against a live Postgres: the deadlock, and its absence.

Two multi-row upserts that touch the same ids in different orders deadlock: each holds a
row the other waits for, and Postgres aborts one. The first test provokes exactly that
with raw statements, so the hazard is on record. The adapter avoids it by writing every
statement's rows in ``_row_order`` instead of serializing inserts behind a per-collection
lock; the second test runs many concurrent, overlapping, differently ordered upserts
through ``create_data_points`` on an HNSW-indexed collection and checks that no deadlock
happened and every tag merged.

No LLM or embedding calls: the embedding engine is a deterministic stub. Skipped unless
``TEST_PGVECTOR_URL`` is set, e.g.:

    TEST_PGVECTOR_URL="postgresql+asyncpg://cognee:cognee@127.0.0.1:5432/cognee_db"
"""

from __future__ import annotations

import asyncio
import os
import random
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from cognee.infrastructure.engine import DataPoint

PG_URL = os.getenv("TEST_PGVECTOR_URL")

pytestmark = [
    pytest.mark.skipif(not PG_URL, reason="TEST_PGVECTOR_URL not set"),
    pytest.mark.asyncio,
]

try:
    from asyncpg import DeadlockDetectedError

    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
        hnsw_index_name,
    )
except ModuleNotFoundError:  # postgres extra not installed
    pytestmark.append(pytest.mark.skip(reason="pgvector extra not installed"))


class _Embeddings:
    """Deterministic embeddings: the same text always maps to the same vector."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def get_vector_size(self) -> int:
        return self.dimensions

    def get_batch_size(self) -> int:
        return 100

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for value in texts:
            rng = random.Random(value)
            vectors.append([rng.uniform(-1, 1) for _ in range(self.dimensions)])
        return vectors


class _Point(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


def _adapter() -> PGVectorAdapter:
    return PGVectorAdapter(
        connection_string=PG_URL, api_key=None, embedding_engine=_Embeddings(16), schema=""
    )


async def _deadlocks(adapter: PGVectorAdapter) -> int:
    """The database's deadlock counter. Backends flush their stats within a second."""
    await asyncio.sleep(1.5)
    async with adapter.engine.connect() as conn:
        return (
            await conn.execute(
                text("SELECT deadlocks FROM pg_stat_database WHERE datname = current_database()")
            )
        ).scalar_one()


@pytest_asyncio.fixture
async def collection():
    adapter = _adapter()
    async with adapter.engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    name = f"ConcurrentLive_{uuid4().hex[:8]}_text"
    await adapter.create_collection(name, _Point)
    yield adapter, name
    async with adapter.engine.begin() as conn:
        await conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
    await adapter.engine.dispose()


async def test_upserts_in_different_orders_deadlock(collection):
    """The hazard: two transactions lock the same two rows in opposite orders."""
    adapter, name = collection
    first, second = sorted(str(uuid4()) for _ in range(2))
    await adapter.create_data_points(
        name, [_Point(id=first, text="a"), _Point(id=second, text="b")]
    )
    upsert = text(
        f"INSERT INTO \"{name}\" (id, payload, vector) VALUES (:id, '{{}}', :vector) "
        "ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload"
    )
    vector = str([0.0] * 16)
    both_hold_one_row = asyncio.Barrier(2)
    deadlocks_before = await _deadlocks(adapter)

    async def transaction(mine: str, theirs: str):
        async with adapter.engine.connect() as conn, conn.begin():
            await conn.execute(upsert, {"id": mine, "vector": vector})
            await both_hold_one_row.wait()
            await conn.execute(upsert, {"id": theirs, "vector": vector})

    outcomes = await asyncio.gather(
        transaction(first, second), transaction(second, first), return_exceptions=True
    )

    failures = [o for o in outcomes if isinstance(o, DBAPIError)]
    assert len(failures) == 1, outcomes
    assert isinstance(failures[0].orig.__cause__, DeadlockDetectedError), failures[0]
    assert await _deadlocks(adapter) == deadlocks_before + 1


async def test_concurrent_overlapping_upserts_do_not_deadlock(collection):
    """Many upserts of the same ids, each in its own order, through the adapter."""
    adapter, name = collection
    assert await adapter.has_collection(name)
    # HNSW takes the concurrent inserts too; the index exists from create_collection.
    async with adapter.engine.connect() as conn:
        indexes = (
            (
                await conn.execute(
                    text("SELECT indexname FROM pg_indexes WHERE tablename = :t"), {"t": name}
                )
            )
            .scalars()
            .all()
        )
    assert hnsw_index_name(name) in indexes

    ids = [str(uuid5(NAMESPACE_URL, f"row-{i}")) for i in range(600)]
    writers = 8
    deadlocks_before = await _deadlocks(adapter)

    async def writer(tag: int):
        order = list(ids)
        random.Random(tag).shuffle(order)
        points = [_Point(id=i, text=f"text {i}", belongs_to_set=[f"set{tag}"]) for i in order]
        await adapter.create_data_points(name, points)

    await asyncio.gather(*[writer(tag) for tag in range(writers)])

    assert await _deadlocks(adapter) == deadlocks_before
    rows = await adapter.retrieve(name, ids)
    assert len(rows) == len(ids)
    expected_tags = sorted(f"set{tag}" for tag in range(writers))
    assert all(sorted(row.payload["belongs_to_set"]) == expected_tags for row in rows)
