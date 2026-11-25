"""Processes that create one pgvector collection at the same time all succeed.

Workers of one pipeline run (see ``cognee.modules.pipelines.execution``) each
create the collections their data needs. ``VECTOR_DB_LOCK`` serializes that
inside one process only, so two processes both pass ``has_collection`` and
Postgres fails the loser's ``CREATE TABLE`` on its catalog. SQLAlchemy raises
that as ``IntegrityError``; ``create_collection`` must retry it and find the
winner's table, or the loser's data item fails.

Adapters that do not share a lock stand in for the processes.

Skipped unless `TEST_PGVECTOR_URL` is set, e.g.:

    TEST_PGVECTOR_URL="postgresql+asyncpg://cognee:cognee@127.0.0.1:5432/cognee_db"
"""

import asyncio
import os
from uuid import uuid4

import pytest

PG_URL = os.getenv("TEST_PGVECTOR_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL,
    reason="TEST_PGVECTOR_URL not set; skipping live pgvector integration tests",
)

try:
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    HAS_PGVECTOR = True
except ModuleNotFoundError:
    HAS_PGVECTOR = False

PROCESSES = 3
COLLECTIONS = 5


class _FakeEmbeddingEngine:
    def get_vector_size(self) -> int:
        return 8


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra not installed")
async def test_every_process_creating_the_same_collection_succeeds():
    adapters = [PGVectorAdapter(PG_URL, None, _FakeEmbeddingEngine()) for _ in range(PROCESSES)]
    names = [f"ConcurrentCreate_{uuid4().hex}_{index}" for index in range(COLLECTIONS)]
    try:
        for name in names:
            # Raises the loser's error if create_collection does not retry it.
            await asyncio.gather(*(adapter.create_collection(name) for adapter in adapters))
            assert await adapters[0].has_collection(name)
    finally:
        for name in names:
            await adapters[0].delete_collection(name)
        for adapter in adapters:
            await adapter.engine.dispose()
