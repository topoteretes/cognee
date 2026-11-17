"""Reuse the binary writer after asyncio.run closes a loop with idle connections."""

import asyncio
import os
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.pool import NullPool

from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

PG_URL = os.getenv("TEST_PGVECTOR_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="TEST_PGVECTOR_URL not set")


class Embeddings:
    def get_vector_size(self):
        return 3


@pytest.mark.parametrize("connections", [1, 2])
def test_reused_adapter_writes_on_the_first_attempt_in_a_new_loop(connections):
    adapter = PGVectorAdapter(PG_URL, api_key=None, embedding_engine=Embeddings())
    # The regression is the new binary pool. Keep SQLAlchemy loop-independent.
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    adapter.engine = create_async_engine(PG_URL, poolclass=NullPool)
    adapter.sessionmaker = async_sessionmaker(adapter.engine, expire_on_commit=False)
    adapter._owns_engine = True
    name = f"PoolLifecycle_{uuid4().hex}"
    first_id, second_id = uuid4(), uuid4()

    async def first():
        await adapter.upsert_raw_vectors(
            name, [{"id": str(first_id), "payload": {}, "vector": [1, 2, 3]}]
        )
        pool = await adapter._get_write_pool()
        held = [await pool.acquire() for _ in range(connections)]
        for connection in held:
            await pool.release(connection)

    async def second():
        await adapter.upsert_raw_vectors(
            name, [{"id": str(second_id), "payload": {}, "vector": [3, 2, 1]}]
        )
        pool = await adapter._get_write_pool()
        async with pool.acquire() as connection:
            assert await connection.fetchval(f'SELECT count(*) FROM "{name}"') == 2

    async def cleanup():
        await adapter.delete_collection(name)
        await adapter.close()

    try:
        asyncio.run(first())
        asyncio.run(second())
        # close() on a third loop must still dispose the SQLAlchemy engine.
        dispose = adapter.engine.dispose
        with patch.object(type(adapter.engine), "dispose", AsyncMock(wraps=dispose)) as disposed:
            asyncio.run(cleanup())
            disposed.assert_awaited_once_with(close=True)
    finally:
        asyncio.run(cleanup())
