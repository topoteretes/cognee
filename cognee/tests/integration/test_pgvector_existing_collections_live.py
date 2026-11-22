"""pgvector ``existing_collections`` against a live Postgres.

The graph retrievers search every collection a ``DataPoint`` subclass could have, most of
which a dataset never creates. ``existing_collections`` answers which ones exist in one
round trip, so the missing ones are skipped instead of each costing a connection. These
tests check the answer in the default schema and in a pinned (shared-database) schema,
where it must match the per-name probe the search used before: a pinned adapter resolves
names through ``search_path = "<schema>, public"``, so its own schema's tables count and
another dataset's schema does not.

No LLM or embedding calls. Skipped unless ``TEST_PGVECTOR_URL`` is set, e.g.:

    TEST_PGVECTOR_URL="postgresql+asyncpg://cognee:cognee@127.0.0.1:5432/cognee_db"
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text

PG_URL = os.getenv("TEST_PGVECTOR_URL")

pytestmark = [
    pytest.mark.skipif(not PG_URL, reason="TEST_PGVECTOR_URL not set"),
    pytest.mark.asyncio,
]

try:
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
    )
except ModuleNotFoundError:  # postgres extra not installed
    pytestmark.append(pytest.mark.skip(reason="pgvector extra not installed"))

DIMENSIONS = 8


class _Embeddings:
    def get_vector_size(self) -> int:
        return DIMENSIONS

    def get_batch_size(self) -> int:
        return 100

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * DIMENSIONS for _ in texts]


@pytest_asyncio.fixture
async def adapter():
    adapter = PGVectorAdapter(
        connection_string=PG_URL, api_key=None, embedding_engine=_Embeddings()
    )
    yield adapter
    await adapter.close()


@pytest_asyncio.fixture
async def pinned_schema(adapter):
    schema = f"ds_existing_{uuid4().hex[:10]}"
    async with adapter.engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    yield schema
    async with adapter.engine.begin() as connection:
        await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))


async def test_reports_only_created_collections_in_given_order(adapter):
    created = [f"IntegExisting_{uuid4().hex[:10]}_text" for _ in range(2)]
    missing = f"IntegMissing_{uuid4().hex[:10]}_text"
    for name in created:
        await adapter.create_collection(name)
    try:
        names = [created[1], missing, created[0]]
        assert await adapter.existing_collections(names) == [created[1], created[0]]
        assert await adapter.existing_collections([missing]) == []
    finally:
        for name in created:
            await adapter.delete_collection(name)


async def test_pinned_schema_matches_the_per_name_probe(adapter, pinned_schema):
    public_only = f"IntegPublic_{uuid4().hex[:10]}_text"
    pinned_only = f"IntegPinned_{uuid4().hex[:10]}_text"
    missing = f"IntegMissing_{uuid4().hex[:10]}_text"
    pinned = PGVectorAdapter(
        connection_string=PG_URL,
        api_key=None,
        embedding_engine=_Embeddings(),
        schema=pinned_schema,
    )
    await adapter.create_collection(public_only)
    try:
        await pinned.create_collection(pinned_only)
        names = [public_only, pinned_only, missing]
        for engine in (pinned, adapter):
            probed = [name for name in names if await engine.has_collection(name)]
            assert await engine.existing_collections(names) == probed
        assert pinned_only in await pinned.existing_collections(names)
        assert pinned_only not in await adapter.existing_collections(names)
    finally:
        await adapter.delete_collection(public_only)
        await pinned.close()
