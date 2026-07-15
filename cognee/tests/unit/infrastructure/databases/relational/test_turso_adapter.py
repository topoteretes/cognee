"""Unit tests for TursoAdapter — the Turso (libSQL) relational backend.

Turso is driven through aiosqlite (a libSQL file is a SQLite file), so the
adapter is a drop-in for the SQLite backend. These tests exercise the local
drop-in against a real temporary libSQL/SQLite file, and the remote
embedded-replica wiring (seed-before-first-use + sync-after-write) with the
libsql driver stubbed, so no network or real Turso database is needed.
"""

import asyncio
import sys
import tempfile
import types
from pathlib import Path

import pytest
from sqlalchemy import text
from unittest.mock import patch

from cognee.infrastructure.databases.relational.sqlalchemy.TursoAdapter import TursoAdapter


def _local_adapter() -> TursoAdapter:
    """A local (embedded) TursoAdapter backed by a fresh temp libSQL/SQLite file."""
    db_path = Path(tempfile.mkdtemp()) / "turso_test.db"
    return TursoAdapter(f"sqlite+aiosqlite:///{db_path}")


class TestTursoAdapterLocal:
    """Local mode is a pure SQLite drop-in: inherit everything, add nothing."""

    def test_is_local_and_uses_sqlite_dialect(self):
        adapter = _local_adapter()
        assert adapter.is_remote is False
        # Inheriting the sqlite dialect is what makes every SQLAlchemyAdapter
        # behavior (and the sqlite-dialect Alembic migrations) apply unchanged.
        assert adapter.engine.dialect.name == "sqlite"

    def test_roundtrip_through_inherited_engine(self):
        adapter = _local_adapter()

        async def scenario():
            async with adapter.engine.begin() as conn:
                await conn.execute(text("CREATE TABLE t (v TEXT)"))
                await conn.execute(text("INSERT INTO t (v) VALUES ('x')"))
            async with adapter.get_async_session() as session:
                rows = (await session.execute(text("SELECT v FROM t"))).scalars().all()
            await adapter.engine.dispose()
            return rows

        assert asyncio.run(scenario()) == ["x"]

    def test_sync_and_write_wrapper_are_noops_locally(self):
        adapter = _local_adapter()
        calls = []

        async def scenario():
            await adapter.sync()  # no-op, must not import/require libsql

            async def fake_write(_self, value):
                calls.append(value)
                return value

            # The write wrapper still runs the underlying method locally; the
            # seed/sync around it are simply no-ops.
            return await adapter._write(fake_write, "written")

        assert asyncio.run(scenario()) == "written"
        assert calls == ["written"]


class TestTursoAdapterRemote:
    """Remote mode adds embedded-replica sync on top of the same aiosqlite engine."""

    @pytest.fixture
    def libsql_stub(self):
        """Stub libsql_experimental so replica sync needs no network or real Turso."""
        calls = {"connect": 0, "sync": 0, "close": 0, "last": None, "fail": False}

        class _Conn:
            def sync(self):
                if calls["fail"]:
                    raise RuntimeError("primary unreachable")
                calls["sync"] += 1

            def close(self):
                calls["close"] += 1

        def _connect(database, sync_url=None, auth_token=None):
            calls["connect"] += 1
            calls["last"] = {
                "database": database,
                "sync_url": sync_url,
                "auth_token": auth_token,
            }
            return _Conn()

        stub = types.ModuleType("libsql_experimental")
        stub.connect = _connect
        with patch.dict(sys.modules, {"libsql_experimental": stub}):
            yield calls

    def _remote_adapter(self) -> TursoAdapter:
        db_path = Path(tempfile.mkdtemp()) / "replica.db"
        return TursoAdapter(
            f"sqlite+aiosqlite:///{db_path}",
            sync_url="libsql://db.turso.io",
            auth_token="tok",
        )

    def test_no_network_in_constructor(self, libsql_stub):
        # __init__ must not touch the network/driver (that would block the loop
        # and hard-fail all DB access when the primary is unreachable).
        adapter = self._remote_adapter()
        assert adapter.is_remote is True
        assert libsql_stub["connect"] == 0

    def test_seeds_replica_once_before_first_use(self, libsql_stub):
        adapter = self._remote_adapter()

        async def scenario():
            async with adapter.get_async_session():
                pass
            async with adapter.get_async_session():
                pass

        asyncio.run(scenario())
        # Seeded exactly once (pull current remote state), passing the wiring through.
        assert libsql_stub["sync"] == 1
        assert libsql_stub["last"]["sync_url"] == "libsql://db.turso.io"
        assert libsql_stub["last"]["auth_token"] == "tok"

    def test_write_syncs_after_the_write(self, libsql_stub):
        adapter = self._remote_adapter()
        order = []

        async def fake_write(_self):
            order.append("write")
            return "done"

        async def scenario():
            return await adapter._write(fake_write)

        result = asyncio.run(scenario())
        assert result == "done"
        # seed (before) + push (after) => two syncs, each its own open/close.
        assert libsql_stub["sync"] == 2
        assert libsql_stub["close"] == 2
        assert order == ["write"]

    def test_sync_failure_is_non_fatal(self, libsql_stub):
        adapter = self._remote_adapter()
        libsql_stub["fail"] = True
        # A transient sync failure must be swallowed so it never breaks a DB op.
        asyncio.run(adapter.sync())
        assert libsql_stub["close"] == 1  # connection still closed despite failure
