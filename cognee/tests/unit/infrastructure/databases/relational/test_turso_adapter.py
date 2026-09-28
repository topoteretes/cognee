"""Unit tests for TursoAdapter — the relational backend on the Turso rewrite engine.

They run against a real temporary Turso database file through cognee's
``sqlite+cognee_turso://`` dialect (pyturso), so they prove the rewrite engine is
the one executing the SQL. No network, no remote Turso database.
"""

import asyncio
import importlib.metadata

import pytest
from sqlalchemy import text

pytest.importorskip("turso", reason="pyturso not installed")

from cognee.infrastructure.databases.relational.sqlalchemy.TursoAdapter import (
    TursoAdapter,
)
from cognee.infrastructure.databases.turso import get_turso_config


def _make_adapter(tmp_path) -> TursoAdapter:
    return TursoAdapter(str(tmp_path / "cognee_db"))


def _run(coro):
    return asyncio.run(coro)


class TestLocalMode:
    def test_uses_cognee_turso_dialect(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        assert adapter.engine.dialect.name == "sqlite"
        assert adapter.engine.dialect.driver == "cognee_turso"
        assert adapter.db_path == str(tmp_path / "cognee_db")
        # Local file: path parsed from the URL exactly like the SQLite branch does.
        assert adapter.engine.url.database == str(tmp_path / "cognee_db")

    def test_rewrite_engine_is_the_executing_engine(self, tmp_path):
        """``turso_version()`` exists only on the Turso rewrite; SQLite has no such function."""
        adapter = _make_adapter(tmp_path)

        async def probe():
            async with adapter.engine.connect() as connection:
                version = (await connection.execute(text("SELECT turso_version()"))).scalar()
                pragmas = {
                    name: (await connection.execute(text(f"PRAGMA {name}"))).scalar()
                    for name in ("journal_mode", "synchronous", "busy_timeout")
                }
            await adapter.engine.dispose()
            return version, pragmas

        version, pragmas = _run(probe())
        assert version, "turso_version() returned nothing — not running on the Turso engine"
        # Every connection PRAGMA of the shared Turso engine policy is in effect.
        config = get_turso_config()
        assert pragmas["journal_mode"] == config.turso_journal_mode
        assert pragmas["synchronous"] in (1, "1", "NORMAL", "normal")
        assert int(pragmas["busy_timeout"]) == config.turso_busy_timeout_ms
        assert importlib.metadata.version("pyturso")

    def test_roundtrip_through_inherited_engine(self, tmp_path):
        adapter = _make_adapter(tmp_path)

        async def roundtrip():
            async with adapter.engine.begin() as connection:
                await connection.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)"))
                await connection.execute(text("INSERT INTO t (v) VALUES (:v)"), {"v": "hello"})
            async with adapter.get_async_session() as session:
                rows = (await session.execute(text("SELECT v FROM t"))).all()
            await adapter.engine.dispose()
            return rows

        assert _run(roundtrip()) == [("hello",)]

    def test_data_persists_across_reopen(self, tmp_path):
        """A second adapter on the same file sees the committed rows (restart semantics)."""
        path = tmp_path / "cognee_db"

        async def write():
            adapter = TursoAdapter(str(path))
            async with adapter.engine.begin() as connection:
                await connection.execute(text("CREATE TABLE t (v TEXT)"))
                await connection.execute(text("INSERT INTO t VALUES ('kept')"))
            await adapter.engine.dispose()

        async def read():
            adapter = TursoAdapter(str(path))
            async with adapter.engine.connect() as connection:
                rows = (await connection.execute(text("SELECT v FROM t"))).all()
            await adapter.engine.dispose()
            return rows

        _run(write())
        assert _run(read()) == [("kept",)]

    def test_create_database_runs_migrations_to_head(self, tmp_path):
        """The full Alembic chain (sqlite dialect) applies on the rewrite engine."""
        adapter = _make_adapter(tmp_path)

        async def migrate():
            await adapter.create_database()
            async with adapter.engine.connect() as connection:
                head = (
                    await connection.execute(text("SELECT version_num FROM alembic_version"))
                ).all()
                tables = (
                    await connection.execute(
                        text("SELECT count(*) FROM sqlite_master WHERE type = 'table'")
                    )
                ).scalar()
            await adapter.engine.dispose()
            return head, tables

        head, tables = _run(migrate())
        assert len(head) == 1
        assert tables > 20

    def test_unsupported_connect_args_are_dropped(self, tmp_path):
        """The SQLite branch passes aiosqlite's ``timeout``; the dialect must swallow it."""
        adapter = TursoAdapter(
            str(tmp_path / "cognee_db"), connect_args={"check_same_thread": False}
        )

        async def probe():
            async with adapter.engine.connect() as connection:
                return (await connection.execute(text("SELECT 1"))).scalar()

        assert _run(probe()) == 1
        _run(adapter.engine.dispose())
