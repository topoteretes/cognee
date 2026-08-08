import asyncio
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import text, NullPool
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)


def _make_sqlite_adapter(tmp_path) -> SQLAlchemyAdapter:
    """Build a real (non-mocked) SQLite adapter backed by a temp file."""
    db_file = (tmp_path / "test.db").as_posix()
    return SQLAlchemyAdapter(f"sqlite+aiosqlite:///{db_file}")


class TestSQLAlchemyAdapterConnectArgs:
    """Verify that connect_args is only passed to create_async_engine when non-empty."""

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_empty_connect_args_not_passed_to_engine(self, mock_create_engine):
        """When connect_args is None/empty, create_async_engine should NOT receive connect_args."""
        mock_create_engine.return_value = MagicMock()

        SQLAlchemyAdapter("postgresql+asyncpg://user:pass@localhost:5432/db")

        _, kwargs = mock_create_engine.call_args
        assert "connect_args" not in kwargs

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_non_empty_connect_args_passed_to_engine(self, mock_create_engine):
        """When connect_args has values, create_async_engine should receive them."""
        mock_create_engine.return_value = MagicMock()

        custom_args = {"sslmode": "require"}
        SQLAlchemyAdapter(
            "postgresql+asyncpg://user:pass@localhost:5432/db",
            connect_args=custom_args,
        )

        _, kwargs = mock_create_engine.call_args
        assert kwargs["connect_args"] == {"sslmode": "require"}

    # event.listens_for needs a real Engine; the mocked engine can't register
    # the SQLite PRAGMA listener, so patch it out for this connect_args check.
    @patch("cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.event")
    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_sqlite_always_passes_connect_args_with_timeout(self, mock_create_engine, mock_event):
        """SQLite should always pass connect_args with at least the timeout default."""
        mock_create_engine.return_value = MagicMock()

        SQLAlchemyAdapter("sqlite+aiosqlite:///tmp/test.db")

        _, kwargs = mock_create_engine.call_args
        assert "timeout" in kwargs["connect_args"]
        assert kwargs["connect_args"]["timeout"] == 120


class TestSQLAlchemyAdapterSqlitePragmas:
    """Regression coverage for issue #2717 (SQLite write deadlock during cognify)."""

    def test_sqlite_connection_enables_wal_and_pragmas(self, tmp_path):
        """Every SQLite connection should come up in WAL mode with the supporting pragmas."""
        adapter = _make_sqlite_adapter(tmp_path)

        async def read_pragmas():
            async with adapter.engine.connect() as conn:
                journal_mode = (await conn.execute(text("PRAGMA journal_mode"))).scalar()
                busy_timeout = (await conn.execute(text("PRAGMA busy_timeout"))).scalar()
            return journal_mode, busy_timeout

        journal_mode, busy_timeout = asyncio.run(read_pragmas())

        assert journal_mode.lower() == "wal"
        assert busy_timeout == 120000

    def test_concurrent_writers_do_not_deadlock(self, tmp_path):
        """Parallel read-then-write transactions must not raise 'database is locked'.

        This is the exact access pattern cognify() produces: many greenlets each
        opening their own connection (NullPool) and updating the same row. In
        rollback-journal mode these deadlock; under WAL + busy_timeout they
        serialize and all commit.
        """
        adapter = _make_sqlite_adapter(tmp_path)

        async def run():
            async with adapter.engine.begin() as conn:
                await conn.execute(
                    text("CREATE TABLE counter (id INTEGER PRIMARY KEY, val INTEGER)")
                )
                await conn.execute(text("INSERT INTO counter (id, val) VALUES (1, 0)"))

            async def increment():
                async with adapter.engine.begin() as conn:
                    current = (
                        await conn.execute(text("SELECT val FROM counter WHERE id = 1"))
                    ).scalar()
                    await conn.execute(
                        text("UPDATE counter SET val = :v WHERE id = 1"),
                        {"v": current + 1},
                    )

            # Raises if any writer hits sqlite3.OperationalError: database is locked.
            await asyncio.gather(*(increment() for _ in range(25)))

            async with adapter.engine.connect() as conn:
                return (await conn.execute(text("SELECT val FROM counter WHERE id = 1"))).scalar()

        final_val = asyncio.run(run())

        # The assertion that matters is that run() completed without a lock error;
        # a committed value confirms the writers actually wrote.
        assert final_val >= 1


class TestSQLAlchemyAdapterSqliteConcurrencyModel:
    """Guards the SQLite concurrency model the #2717 fix depends on.

    The lock fix is NullPool (connection-per-greenlet) + WAL + busy_timeout. Two
    tempting "optimizations" both look like they fix the lock but break cognee:

      * StaticPool collapses every coroutine onto one shared connection, so their
        transactions merge -> dirty reads and one greenlet's rollback discards
        another's committed writes (silent corruption).
      * A size-1 QueuePool (max_overflow=0) serializes checkouts, so the adapter's
        own nested checkout (get_table() opens a second connection inside
        insert_data()/delete_*()) deadlocks waiting on the connection it holds.

    These tests fail loudly if either change is ever introduced.
    """

    def test_sqlite_uses_nullpool(self, tmp_path):
        """SQLite must use NullPool: a fresh connection (and transaction) per checkout."""
        adapter = _make_sqlite_adapter(tmp_path)
        assert isinstance(adapter.engine.pool, NullPool)

    def test_concurrent_sessions_are_transaction_isolated(self, tmp_path):
        """A concurrent session must NOT see another session's uncommitted row.

        This is the StaticPool detector: under a shared connection the reader sees
        the writer's uncommitted insert (a dirty read). With NullPool each session
        owns its connection/transaction, so the read returns nothing.
        """
        adapter = _make_sqlite_adapter(tmp_path)

        async def run():
            async with adapter.engine.begin() as conn:
                await conn.execute(text("CREATE TABLE iso (name TEXT)"))

            writer_inserted = asyncio.Event()
            reader_done = asyncio.Event()
            seen = {}

            async def writer():
                async with adapter.get_async_session() as session:
                    await session.execute(text("INSERT INTO iso (name) VALUES ('A')"))
                    await session.flush()  # written to the connection, NOT committed
                    writer_inserted.set()
                    await reader_done.wait()
                    await session.rollback()

            async def reader():
                await writer_inserted.wait()
                async with adapter.get_async_session() as session:
                    rows = (await session.execute(text("SELECT name FROM iso"))).all()
                    seen["rows"] = [r[0] for r in rows]
                reader_done.set()

            await asyncio.gather(writer(), reader())
            return seen["rows"]

        assert asyncio.run(run()) == []  # no dirty read -> transactions are isolated

    def test_nested_checkout_does_not_deadlock(self, tmp_path):
        """Opening a second connection while the first is held must not deadlock.

        This mirrors insert_data()/delete_*() calling get_table() (which opens its
        own engine.begin()) inside an already-open connection. NullPool just opens
        another connection; a serialized size-1 pool would hang here.
        """
        adapter = _make_sqlite_adapter(tmp_path)

        async def run():
            async with adapter.engine.begin() as outer:
                await outer.execute(text("SELECT 1"))
                async with adapter.engine.begin() as inner:
                    return (await inner.execute(text("SELECT 2"))).scalar()

        assert asyncio.run(asyncio.wait_for(run(), timeout=15)) == 2


class TestGetAsyncSessionCancellation:
    """
    get_async_session must invalidate the connection when its task is cancelled.

    A cancelled request can leave the DBAPI connection inside an open
    transaction. Returning it to the pool leaks an "idle in transaction"
    backend until the pool ceiling is hit and every request fails. On
    cancellation the session must be invalidated (connection dropped), and
    only on cancellation. See issue #4197.

    These tests assert the invalidate-on-cancel contract on the three paths
    (happy, ordinary exception, cancellation). They run on SQLite, which has
    no QueuePool or "idle in transaction" state, so they guard the behaviour
    that fixes the leak rather than reproducing the pool exhaustion itself.
    The end-to-end Postgres reproduction is the curl-abort + pg_stat_activity
    check documented in the issue.
    """

    def _spy_adapter(self, monkeypatch):
        adapter = SQLAlchemyAdapter("sqlite+aiosqlite:///:memory:")
        calls = {"invalidate": 0}
        original = AsyncSession.invalidate

        async def spy(self):
            calls["invalidate"] += 1
            return await original(self)

        monkeypatch.setattr(AsyncSession, "invalidate", spy)
        return adapter, calls

    @pytest.mark.asyncio
    async def test_happy_path_does_not_invalidate(self, monkeypatch):
        adapter, calls = self._spy_adapter(monkeypatch)
        async with adapter.get_async_session() as session:
            await session.execute(text("SELECT 1"))
        assert calls["invalidate"] == 0

    @pytest.mark.asyncio
    async def test_ordinary_exception_does_not_invalidate(self, monkeypatch):
        adapter, calls = self._spy_adapter(monkeypatch)
        with pytest.raises(ValueError):
            async with adapter.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                raise ValueError("boom")
        assert calls["invalidate"] == 0

    @pytest.mark.asyncio
    async def test_cancellation_invalidates_and_reraises(self, monkeypatch):
        adapter, calls = self._spy_adapter(monkeypatch)
        with pytest.raises(asyncio.CancelledError):
            async with adapter.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                raise asyncio.CancelledError()
        assert calls["invalidate"] == 1

    @pytest.mark.asyncio
    async def test_outer_task_cancellation_invalidates(self, monkeypatch):
        adapter, calls = self._spy_adapter(monkeypatch)
        started = asyncio.Event()

        async def worker():
            async with adapter.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                started.set()
                await asyncio.sleep(10)

        task = asyncio.create_task(worker())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls["invalidate"] == 1

    @pytest.mark.asyncio
    async def test_invalidate_failure_does_not_mask_cancellation(self, monkeypatch):
        adapter, calls = self._spy_adapter(monkeypatch)

        async def failing_invalidate(self):
            calls["invalidate"] += 1
            raise RuntimeError("invalidate failed")

        monkeypatch.setattr(AsyncSession, "invalidate", failing_invalidate)
        with pytest.raises(asyncio.CancelledError):
            async with adapter.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                raise asyncio.CancelledError()
        assert calls["invalidate"] == 1


class TestSQLAlchemyAdapterSqlitePoolArgs:
    """POOL_ARGS is honored on the sqlite branch (issue #4328).

    Defaults are unchanged: with no pool_args the sqlite engine keeps NullPool.
    ``poolclass: "nullpool"`` is normalized to the class, matching the
    server-database branch, so one POOL_ARGS value means the same thing on
    every backend.
    """

    def test_default_stays_nullpool(self, tmp_path):
        adapter = _make_sqlite_adapter(tmp_path)
        assert isinstance(adapter.engine.pool, NullPool)

    def test_sizing_keys_opt_into_a_bounded_pool(self, tmp_path):
        db_file = (tmp_path / "pooled.db").as_posix()
        adapter = SQLAlchemyAdapter(
            f"sqlite+aiosqlite:///{db_file}",
            pool_args={"pool_size": 3, "max_overflow": 7, "pool_timeout": 11},
        )
        pool = adapter.engine.pool
        assert not isinstance(pool, NullPool)
        assert pool.size() == 3
        assert pool._max_overflow == 7
        assert pool._timeout == 11

    def test_nullpool_string_is_normalized(self, tmp_path):
        db_file = (tmp_path / "nullpool.db").as_posix()
        adapter = SQLAlchemyAdapter(
            f"sqlite+aiosqlite:///{db_file}",
            pool_args={"poolclass": "nullpool"},
        )
        assert isinstance(adapter.engine.pool, NullPool)

    def test_nullpool_with_sizing_keys_fails_fast(self, tmp_path):
        """NullPool takes no sizing arguments; the contradiction must error
        (same as the server-database branch), never be silently resolved."""
        db_file = (tmp_path / "conflict.db").as_posix()
        with pytest.raises(TypeError):
            SQLAlchemyAdapter(
                f"sqlite+aiosqlite:///{db_file}",
                pool_args={"poolclass": "nullpool", "pool_size": 3},
            )

    def test_pooled_sqlite_still_executes(self, tmp_path):
        db_file = (tmp_path / "exec.db").as_posix()
        adapter = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{db_file}", pool_args={"pool_size": 2})

        async def run():
            async with adapter.get_async_session() as session:
                return (await session.execute(text("select 41 + 1"))).scalar()

        assert asyncio.run(run()) == 42
