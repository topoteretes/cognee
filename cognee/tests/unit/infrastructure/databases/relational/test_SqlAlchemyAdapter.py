import asyncio
from unittest.mock import patch, MagicMock

from sqlalchemy import text

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
        assert kwargs["connect_args"]["timeout"] == 30


class TestSQLAlchemyAdapterSqlitePragmas:
    """Regression coverage for issue #2717 (SQLite write deadlock during cognify)."""

    def test_sqlite_connection_enables_wal_and_pragmas(self, tmp_path):
        """Every SQLite connection should come up in WAL mode with the supporting pragmas."""
        adapter = _make_sqlite_adapter(tmp_path)

        async def read_pragmas():
            async with adapter.engine.connect() as conn:
                journal_mode = (await conn.execute(text("PRAGMA journal_mode"))).scalar()
                busy_timeout = (await conn.execute(text("PRAGMA busy_timeout"))).scalar()
                foreign_keys = (await conn.execute(text("PRAGMA foreign_keys"))).scalar()
            return journal_mode, busy_timeout, foreign_keys

        journal_mode, busy_timeout, foreign_keys = asyncio.run(read_pragmas())

        assert journal_mode.lower() == "wal"
        assert busy_timeout == 30000
        assert foreign_keys == 1

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
