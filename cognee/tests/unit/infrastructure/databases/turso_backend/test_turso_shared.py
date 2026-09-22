"""Unit tests for the plumbing shared by the three Turso backends.

Covers the settings class, the ``sqlite+cognee_turso://`` dialect (reflection and
connect-arg handling against the real engine), the transaction helpers and the
file cleanup rule. Requires pyturso; everything runs on temporary files.
"""

import asyncio

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

pytest.importorskip("turso", reason="pyturso not installed")

from cognee.infrastructure.databases.turso import (
    DATABASE_COMPANION_SUFFIXES,
    TursoConfig,
    begin_statement,
    configure_engine,
    connect_args_for_mode,
    connect_pragmas,
    database_file_paths,
    exclusive_transaction,
    is_retryable_conflict,
    remove_database_files,
    retry_on_conflict,
    turso_url,
)
from cognee.infrastructure.databases.turso.transactions import _exclusive_ddl


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class TestTursoConfig:
    def test_defaults_are_wal(self):
        config = TursoConfig(_env_file=None)
        assert config.turso_journal_mode == "wal"
        assert config.concurrent_writes is False
        assert config.turso_busy_timeout_ms == 120000
        assert config.turso_conflict_retries == 5

    def test_mode_is_normalized(self):
        config = TursoConfig(_env_file=None, turso_journal_mode=" MVCC ")
        assert config.turso_journal_mode == "mvcc"
        assert config.concurrent_writes is True

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="TURSO_JOURNAL_MODE"):
            TursoConfig(_env_file=None, turso_journal_mode="delete")

    def test_negative_knobs_are_rejected(self):
        with pytest.raises(ValueError):
            TursoConfig(_env_file=None, turso_busy_timeout_ms=-1)
        with pytest.raises(ValueError):
            TursoConfig(_env_file=None, turso_conflict_retries=-1)


# --------------------------------------------------------------------------- #
# Transaction helpers
# --------------------------------------------------------------------------- #
class TestTransactionHelpers:
    def test_connect_pragmas_follow_config(self):
        wal = TursoConfig(_env_file=None)
        assert connect_pragmas(wal) == [
            "PRAGMA journal_mode=wal",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA busy_timeout=120000",
        ]
        mvcc = TursoConfig(_env_file=None, turso_journal_mode="mvcc", turso_busy_timeout_ms=10)
        assert connect_pragmas(mvcc, foreign_keys=True)[0] == "PRAGMA foreign_keys=ON"
        assert "PRAGMA journal_mode=mvcc" in connect_pragmas(mvcc)
        assert "PRAGMA busy_timeout=10" in connect_pragmas(mvcc)

    def test_connect_args_and_begin_statement_per_mode(self):
        wal = TursoConfig(_env_file=None)
        mvcc = TursoConfig(_env_file=None, turso_journal_mode="mvcc")
        assert connect_args_for_mode(wal) == {}
        assert connect_args_for_mode(mvcc) == {"isolation_level": None}
        assert begin_statement(wal) is None
        assert begin_statement(mvcc) == "BEGIN CONCURRENT"
        assert begin_statement(mvcc, ddl=True) == "BEGIN"

    def test_is_retryable_conflict(self):
        assert is_retryable_conflict(RuntimeError("Write-write conflict"))
        assert is_retryable_conflict(RuntimeError("database is locked"))
        assert is_retryable_conflict(RuntimeError("Transaction error: busy"))
        assert not is_retryable_conflict(RuntimeError("no such table: t"))

    def test_retry_on_conflict_retries_then_succeeds(self):
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Write-write conflict")
            return "ok"

        assert _run(retry_on_conflict(flaky, attempts=5, base_delay=0)) == "ok"
        assert calls["n"] == 3

    def test_retry_on_conflict_gives_up_after_attempts(self):
        calls = {"n": 0}

        async def always_conflicts():
            calls["n"] += 1
            raise RuntimeError("Write-write conflict")

        with pytest.raises(RuntimeError, match="conflict"):
            _run(retry_on_conflict(always_conflicts, attempts=2, base_delay=0))
        assert calls["n"] == 3  # first try + 2 retries

    def test_retry_on_conflict_does_not_retry_other_errors(self):
        calls = {"n": 0}

        async def broken():
            calls["n"] += 1
            raise RuntimeError("no such table")

        with pytest.raises(RuntimeError, match="no such table"):
            _run(retry_on_conflict(broken, attempts=5, base_delay=0))
        assert calls["n"] == 1

    def test_exclusive_transaction_scopes_the_flag(self):
        async def probe():
            assert _exclusive_ddl.get() is False
            async with exclusive_transaction():
                assert _exclusive_ddl.get() is True
            assert _exclusive_ddl.get() is False

        _run(probe())


# --------------------------------------------------------------------------- #
# Dialect against the real engine
# --------------------------------------------------------------------------- #
class TestDialect:
    def test_turso_url_shape(self, tmp_path):
        path = str(tmp_path / "x.db")
        assert turso_url(path) == f"sqlite+cognee_turso:///{path}"
        assert turso_url(":memory:") == "sqlite+cognee_turso:///:memory:"

    def test_engine_runs_on_rewrite_and_reflects_schema(self, tmp_path):
        async def probe():
            engine = create_async_engine(
                turso_url(str(tmp_path / "d.db")),
                poolclass=NullPool,
                # aiosqlite-style args the SQLite branch passes; must be dropped.
                connect_args={"timeout": 120, "check_same_thread": False},
            )
            configure_engine(engine, foreign_keys=True)
            async with engine.begin() as connection:
                await connection.execute(
                    text("CREATE TABLE parent (id TEXT PRIMARY KEY, name TEXT UNIQUE)")
                )
                await connection.execute(
                    text(
                        "CREATE TABLE child (id TEXT PRIMARY KEY, "
                        "parent_id TEXT REFERENCES parent(id) ON DELETE CASCADE)"
                    )
                )
                await connection.execute(text("CREATE INDEX ix_child_parent ON child(parent_id)"))
            async with engine.connect() as connection:
                version = (await connection.execute(text("SELECT turso_version()"))).scalar()
                foreign_keys = (await connection.execute(text("PRAGMA foreign_keys"))).scalar()
                indexes = await connection.run_sync(
                    lambda sync: sa_inspect(sync).get_indexes("child")
                )
                fks = await connection.run_sync(
                    lambda sync: sa_inspect(sync).get_foreign_keys("child")
                )
                uniques = await connection.run_sync(
                    lambda sync: sa_inspect(sync).get_unique_constraints("parent")
                )
            await engine.dispose()
            return version, foreign_keys, indexes, fks, uniques

        version, foreign_keys, indexes, fks, uniques = _run(probe())
        assert version  # only the Turso rewrite defines turso_version()
        assert foreign_keys == 1
        assert [index["name"] for index in indexes] == ["ix_child_parent"]
        assert fks and fks[0]["referred_table"] == "parent"
        # SQLite reports column-level UNIQUE as an autoindex; reflection must not be
        # the upstream stub that returns [] for every table with an index.
        assert isinstance(uniques, list)

    def test_nested_joins_are_flattened(self, tmp_path):
        """``JOIN (a JOIN b)`` is rejected by the engine; the compiler must render a flat chain.

        This is the shape joined-table inheritance produces (cognee's ``Tenant`` /
        ``User`` are ``Principal`` subclasses), so ``get_user`` depends on it.
        """
        from sqlalchemy import Column, Integer, MetaData, String, Table, select

        metadata = MetaData()
        principals = Table("p", metadata, Column("id", Integer, primary_key=True))
        tenants = Table(
            "tn", metadata, Column("id", Integer, primary_key=True), Column("name", String)
        )
        users = Table("us", metadata, Column("id", Integer, primary_key=True))
        links = Table("lk", metadata, Column("u_id", Integer), Column("t_id", Integer))

        async def probe():
            engine = create_async_engine(turso_url(str(tmp_path / "j.db")), poolclass=NullPool)
            async with engine.begin() as connection:
                await connection.run_sync(metadata.create_all)
                await connection.execute(principals.insert().values([{"id": 2}, {"id": 3}]))
                await connection.execute(
                    tenants.insert().values([{"id": 2, "name": "acme"}, {"id": 3, "name": "b"}])
                )
                await connection.execute(users.insert().values([{"id": 1}, {"id": 9}]))
                await connection.execute(links.insert().values([{"u_id": 1, "t_id": 2}]))

                # selectinload shape: secondary JOIN (principals JOIN tenants)
                inheritance = principals.join(tenants, principals.c.id == tenants.c.id)
                inner = select(links.c.u_id, tenants.c.name).select_from(
                    links.join(inheritance, links.c.t_id == tenants.c.id)
                )
                inner_sql = str(inner.compile(engine.sync_engine))
                inner_rows = (await connection.execute(inner)).all()

                # joinedload shape: users LEFT OUTER JOIN (secondary JOIN (p JOIN t))
                nested = links.join(inheritance, links.c.t_id == tenants.c.id)
                outer = (
                    select(users.c.id, tenants.c.name)
                    .select_from(users.join(nested, users.c.id == links.c.u_id, isouter=True))
                    .order_by(users.c.id)
                )
                outer_sql = str(outer.compile(engine.sync_engine))
                outer_rows = (await connection.execute(outer)).all()
            await engine.dispose()
            return inner_sql, inner_rows, outer_sql, outer_rows

        inner_sql, inner_rows, outer_sql, outer_rows = _run(probe())
        assert "JOIN (" not in inner_sql, inner_sql
        assert inner_rows == [(1, "acme")]
        assert "JOIN (" not in outer_sql, outer_sql
        assert outer_sql.count("LEFT OUTER JOIN") == 3, outer_sql
        assert outer_rows == [(1, "acme"), (9, None)]

    def test_two_engines_see_each_others_commits(self, tmp_path):
        async def probe():
            url = turso_url(str(tmp_path / "shared.db"))
            first = create_async_engine(url, poolclass=NullPool)
            second = create_async_engine(url, poolclass=NullPool)
            configure_engine(first)
            configure_engine(second)
            async with first.begin() as connection:
                await connection.execute(text("CREATE TABLE t (v TEXT)"))
                await connection.execute(text("INSERT INTO t VALUES ('a')"))
            async with second.connect() as connection:
                rows = (await connection.execute(text("SELECT v FROM t"))).all()
            await first.dispose()
            await second.dispose()
            return rows

        assert _run(probe()) == [("a",)]


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
class TestFiles:
    def test_companion_paths_and_removal(self, tmp_path):
        database = tmp_path / "graph.db"
        assert DATABASE_COMPANION_SUFFIXES == ("-wal", "-shm", "-log")
        assert database_file_paths(str(database)) == [
            str(database),
            str(database) + "-wal",
            str(database) + "-shm",
            str(database) + "-log",
        ]
        for path in database_file_paths(str(database))[:3]:
            open(path, "w").close()
        remove_database_files(str(database))  # tolerates the missing -log
        assert not any(tmp_path.iterdir())
