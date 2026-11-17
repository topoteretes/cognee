"""A cached adapter can retire a pool whose owning event loop has closed."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter
from cognee.infrastructure.locks.loop_agnostic_lock import LoopAgnosticLock


@pytest.mark.asyncio
async def test_closed_loop_pool_is_detached_before_termination_and_replaced(monkeypatch):
    adapter = object.__new__(PGVectorAdapter)
    old_loop = asyncio.new_event_loop()
    old_loop.close()

    def terminate():
        assert adapter._write_pool is None
        assert adapter._write_pool_loop is None
        raise RuntimeError("Event loop is closed")

    adapter._write_pool = SimpleNamespace(terminate=terminate)
    adapter._write_pool_loop = old_loop
    adapter._write_pool_lock = LoopAgnosticLock()
    adapter._write_connect_args = {}
    adapter._asyncpg_dsn = lambda: "postgresql://unused"
    fresh = object()
    create = AsyncMock(return_value=fresh)
    monkeypatch.setattr("asyncpg.create_pool", create)
    assert await adapter._get_write_pool() is fresh
    assert await adapter._get_write_pool() is fresh
    create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", [True, False])
async def test_close_always_disposes_owned_engine_after_pool_termination(closed):
    adapter = object.__new__(PGVectorAdapter)
    old_loop = asyncio.new_event_loop()
    if closed:
        old_loop.close()
    adapter._write_pool = SimpleNamespace(terminate=Mock(side_effect=RuntimeError("terminate")))
    adapter._write_pool_loop = old_loop
    adapter._owns_engine = True
    adapter.engine = SimpleNamespace(dispose=AsyncMock())
    try:
        if closed:
            await adapter.close()
        else:
            with pytest.raises(RuntimeError, match="terminate"):
                await adapter.close()
        assert adapter._write_pool is None
        assert adapter._write_pool_loop is None
        adapter.engine.dispose.assert_awaited_once_with(close=True)
    finally:
        old_loop.close()
