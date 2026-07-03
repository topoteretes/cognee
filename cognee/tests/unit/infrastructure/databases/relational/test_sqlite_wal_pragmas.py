"""
Unit tests for SQLite WAL-mode pragma injection in SQLAlchemyAdapter.

Regression coverage for issue #2717: cognify() SQLite deadlock caused by
intra-process greenlet contention. Fix mirrors the identical pattern already
present in SqlCacheAdapter (lines 140-146).
"""
from unittest.mock import patch, MagicMock, call
from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)


class TestSQLAlchemyAdapterSQLitePragmas:

    @patch("cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.sa_event")
    @patch("cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine")
    def test_sqlite_wal_pragmas_registered(self, mock_create_engine, mock_sa_event):
        """
        SQLAlchemyAdapter must register a connect event listener on the SQLite
        sync_engine. Fix for intra-process greenlet deadlock in cognify() (issue #2717)
        — mirrors the identical pattern in SqlCacheAdapter.
        """
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_sa_event.listens_for.return_value = lambda fn: fn

        SQLAlchemyAdapter("sqlite+aiosqlite:///tmp/test.db")

        mock_sa_event.listens_for.assert_called_once_with(
            mock_engine.sync_engine, "connect"
        )

    @patch("cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.sa_event")
    @patch("cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine")
    def test_sqlite_pragmas_execute_on_connect(self, mock_create_engine, mock_sa_event):
        """
        The registered handler must issue journal_mode=WAL, busy_timeout=30000,
        and synchronous=NORMAL in that order.
        """
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        captured = {}

        def fake_listens_for(target, event_name):
            def decorator(fn):
                captured["fn"] = fn
                return fn
            return decorator

        mock_sa_event.listens_for.side_effect = fake_listens_for
        SQLAlchemyAdapter("sqlite+aiosqlite:///tmp/test.db")

        assert "fn" in captured, "No event listener was registered."

        mock_cursor = MagicMock()
        mock_dbapi_conn = MagicMock()
        mock_dbapi_conn.cursor.return_value = mock_cursor
        captured["fn"](mock_dbapi_conn, None)

        mock_cursor.execute.assert_has_calls([
            call("PRAGMA journal_mode=WAL"),
            call("PRAGMA busy_timeout=30000"),
            call("PRAGMA synchronous=NORMAL"),
        ])
        mock_cursor.close.assert_called_once()

    @patch("cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.sa_event")
    @patch("cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine")
    def test_postgres_does_not_set_sqlite_pragmas(self, mock_create_engine, mock_sa_event):
        """PostgreSQL connections must never trigger the SQLite pragma handler."""
        mock_create_engine.return_value = MagicMock()

        SQLAlchemyAdapter("postgresql+asyncpg://user:pass@localhost:5432/db")

        mock_sa_event.listens_for.assert_not_called()