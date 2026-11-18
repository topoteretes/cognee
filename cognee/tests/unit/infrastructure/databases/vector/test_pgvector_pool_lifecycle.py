"""A cached adapter can retire a pool whose owning event loop has closed."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
    _ACCESS_CONTROL_DEFAULT_POOL_ARGS,
    PGVectorAdapter,
    _write_pool_size,
)
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
    adapter._write_pool_max_size = 7
    adapter._asyncpg_dsn = lambda: "postgresql://unused"
    fresh = object()
    create = AsyncMock(return_value=fresh)
    monkeypatch.setattr("asyncpg.create_pool", create)
    assert await adapter._get_write_pool() is fresh
    assert await adapter._get_write_pool() is fresh
    create.assert_awaited_once()
    assert create.await_args.kwargs["max_size"] == 7


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


@pytest.mark.asyncio
async def test_write_waits_for_a_connection_only_up_to_the_pool_timeout():
    """A write asks the binary write pool for a connection with the adapter's
    pool_timeout, so a full pool fails the write instead of waiting forever."""
    adapter = object.__new__(PGVectorAdapter)
    adapter._write_pool_acquire_timeout = 360
    adapter._qualified_table = lambda name: name
    connection = SimpleNamespace(transaction=Mock(), executemany=AsyncMock())
    connection.transaction.return_value.__aenter__ = AsyncMock()
    connection.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    acquired = Mock()
    acquired.__aenter__ = AsyncMock(return_value=connection)
    acquired.__aexit__ = AsyncMock(return_value=False)
    pool = SimpleNamespace(acquire=Mock(return_value=acquired))
    adapter._get_write_pool = AsyncMock(return_value=pool)

    await adapter._write_rows(
        "c", [{"id": "1", "payload": {}, "vector": [0.0]}], lambda table, name: "SQL"
    )

    pool.acquire.assert_called_once_with(timeout=360)


def test_write_pool_ceiling_is_the_sqlalchemy_pools_ceiling():
    """One setting sizes both pools: pool_size + max_overflow of the adapter's pool args,
    with the SQLAlchemy adapter's defaults for a key that is not given."""
    assert _write_pool_size({"pool_size": 3, "max_overflow": 4}) == 7
    assert _write_pool_size(_ACCESS_CONTROL_DEFAULT_POOL_ARGS) == 12
    assert _write_pool_size({}) == 40
    assert _write_pool_size({"pool_size": 1, "max_overflow": 0, "pool_recycle": 60}) == 1
    assert _write_pool_size({"poolclass": "nullpool"}) == 40


@pytest.mark.parametrize("pool_args", [{"pool_size": 0}, {"max_overflow": -1}])
def test_unbounded_pool_args_are_refused(pool_args):
    with pytest.raises(ValueError, match="bounded connection pool"):
        _write_pool_size(pool_args)
