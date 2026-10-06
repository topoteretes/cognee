"""Unit tests for the plumbing shared by the three Turso backends.

Covers the settings class, the ``sqlite+cognee_turso://`` dialect (reflection and
connect-arg handling against the real engine), the transaction helpers and the
file cleanup rule. Requires pyturso; everything runs on temporary files.
"""

import asyncio

import pytest
from sqlalchemy import event, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import DBAPIError
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
    explain_file_in_use,
    is_locked_by_another_process,
    is_retryable_conflict,
    remove_database_files,
    retry_on_conflict,
    turso_url,
    write_transaction,
)
from cognee.infrastructure.databases.turso.transactions import _exclusive_ddl


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def no_turso_env(monkeypatch):
    """Keep TURSO_* from the environment (e.g. a CI mvcc pass) out of explicit configs."""
    for name in ("TURSO_JOURNAL_MODE", "TURSO_BUSY_TIMEOUT_MS", "TURSO_CONFLICT_RETRIES"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("no_turso_env")
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

    def test_wal_only_pins_journal_mode_and_keeps_other_knobs(self):
        mvcc = TursoConfig(_env_file=None, turso_journal_mode="mvcc", turso_busy_timeout_ms=10)
        pinned = mvcc.wal_only()
        assert pinned.turso_journal_mode == "wal"
        assert pinned.concurrent_writes is False
        assert pinned.turso_busy_timeout_ms == 10
        assert mvcc.concurrent_writes is True  # the source config is untouched


# --------------------------------------------------------------------------- #
# Transaction helpers
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("no_turso_env")
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
        # Read-modify-write transactions: the read must already hold the write lock.
        assert begin_statement(wal, write=True) == "BEGIN IMMEDIATE"
        assert begin_statement(mvcc, write=True) == "BEGIN CONCURRENT"
        assert connect_args_for_mode(wal, immediate_writes=True) == {"isolation_level": None}

    @pytest.mark.parametrize(
        "message",
        [
            "Write-write conflict",
            "(turso.lib.DatabaseError) Write-write conflict",
            "database is locked",
            "(turso.lib.OperationalError) database is locked",
            "Transaction error: busy",
            "Busy snapshot",
        ],
    )
    def test_is_retryable_conflict_matches_engine_contention_messages(self, message):
        assert is_retryable_conflict(RuntimeError(message))

    @pytest.mark.parametrize(
        "message",
        [
            "no such table: t",
            "UNIQUE constraint failed: graph_node.id",
            "FOREIGN KEY constraint failed",
            "Parse error: ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint",
            "Transaction error: Concurrent transaction mode is only supported when MVCC is enabled",
            "Transaction error: cannot commit - no transaction is active",
            "conflicting declarations",
        ],
    )
    def test_is_retryable_conflict_rejects_deterministic_errors(self, message):
        """A generic 'conflict' substring is not enough: these must fail immediately."""
        assert not is_retryable_conflict(RuntimeError(message))

    def test_is_retryable_conflict_reads_the_driver_error_not_the_statement(self):
        """SQLAlchemy's str() includes SQL and params; user text must not trigger a retry."""
        constraint = DBAPIError(
            "INSERT INTO t (v) VALUES (?)",
            ("chunk text: the database is locked again",),
            RuntimeError("UNIQUE constraint failed: t.v"),
        )
        assert "database is locked" in str(constraint).lower()
        assert not is_retryable_conflict(constraint)

        conflict = DBAPIError("UPDATE t SET v = ?", ("x",), RuntimeError("Write-write conflict"))
        assert is_retryable_conflict(conflict)

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

    def test_write_transaction_begins_immediate_in_wal(self, tmp_path):
        """With immediate_writes, writes open BEGIN IMMEDIATE and reads stay deferred."""
        config = TursoConfig(_env_file=None)
        seen: list[str] = []

        async def probe():
            engine = create_async_engine(
                turso_url(str(tmp_path / "w.db")),
                poolclass=NullPool,
                connect_args=connect_args_for_mode(config, immediate_writes=True),
            )
            configure_engine(engine, config=config, immediate_writes=True)

            @event.listens_for(engine.sync_engine, "before_cursor_execute")
            def record(conn, cursor, statement, *args):
                if statement.startswith("BEGIN"):
                    seen.append(statement)

            async with engine.begin() as connection:
                await connection.execute(text("CREATE TABLE t (v TEXT)"))
            async with write_transaction(), engine.begin() as connection:
                await connection.execute(text("INSERT INTO t VALUES ('a')"))
            async with engine.connect() as connection:
                rows = (await connection.execute(text("SELECT v FROM t"))).all()
            await engine.dispose()
            return rows

        assert _run(probe()) == [("a",)]
        assert seen == ["BEGIN", "BEGIN IMMEDIATE", "BEGIN"]

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
            # DDL under mvcc (TURSO_JOURNAL_MODE) needs a plain BEGIN; a no-op in wal.
            async with exclusive_transaction(), engine.begin() as connection:
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
        from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table, select

        metadata = MetaData()
        principals = Table("p", metadata, Column("id", Integer, primary_key=True))
        tenants = Table(
            "tn",
            metadata,
            Column("id", Integer, ForeignKey("p.id"), primary_key=True),
            Column("name", String),
        )
        users = Table("us", metadata, Column("id", Integer, primary_key=True))
        links = Table(
            "lk",
            metadata,
            Column("u_id", Integer, ForeignKey("us.id")),
            Column("t_id", Integer, ForeignKey("tn.id")),
        )

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

    def test_flattened_outer_join_null_and_missing_fk_semantics(self, tmp_path):
        """Pin the outer-join equivalence the compiler relies on, including its one divergence.

        ``users LEFT OUTER JOIN (links JOIN (p JOIN t))``: a user with no link yields
        NULLs (identical to the nested form); a link to a ``t`` row whose inheritance
        parent ``p`` is missing (broken integrity) yields the link/``t`` columns with
        NULL ``p`` columns where the nested form would yield all NULLs. The foreign keys
        are declared (that is what lets the compiler flatten), the data breaks one.
        """
        from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table, select

        metadata = MetaData()
        principals = Table("p", metadata, Column("id", Integer, primary_key=True))
        tenants = Table(
            "tn",
            metadata,
            Column("id", Integer, ForeignKey("p.id"), primary_key=True),
            Column("name", String),
        )
        users = Table("us", metadata, Column("id", Integer, primary_key=True))
        links = Table(
            "lk",
            metadata,
            Column("u_id", Integer, ForeignKey("us.id")),
            Column("t_id", Integer, ForeignKey("tn.id")),
        )

        async def probe():
            engine = create_async_engine(turso_url(str(tmp_path / "j2.db")), poolclass=NullPool)
            async with engine.begin() as connection:
                await connection.run_sync(metadata.create_all)
                await connection.execute(principals.insert().values([{"id": 2}]))
                # tenant 3 has no principal row: broken inheritance integrity
                await connection.execute(
                    tenants.insert().values(
                        [{"id": 2, "name": "acme"}, {"id": 3, "name": "orphan"}]
                    )
                )
                await connection.execute(users.insert().values([{"id": 1}, {"id": 5}, {"id": 9}]))
                await connection.execute(
                    links.insert().values([{"u_id": 1, "t_id": 2}, {"u_id": 5, "t_id": 3}])
                )
                inheritance = principals.join(tenants, principals.c.id == tenants.c.id)
                nested = links.join(inheritance, links.c.t_id == tenants.c.id)
                statement = (
                    select(users.c.id, principals.c.id, tenants.c.name)
                    .select_from(users.join(nested, users.c.id == links.c.u_id, isouter=True))
                    .order_by(users.c.id)
                )
                rows = (await connection.execute(statement)).all()
            await engine.dispose()
            return rows

        rows = _run(probe())
        assert rows[0] == (1, 2, "acme")  # full match
        assert rows[1] == (5, None, "orphan")  # divergence: nested form gives (5, None, None)
        assert rows[2] == (9, None, None)  # no link at all: NULLs, identical to nested form

    def test_outer_group_steps_must_be_total(self, tmp_path):
        """A later step of an outer-joined group must follow a foreign key, nothing else.

        Each unsafe shape runs nested on stdlib sqlite3, where a principal never comes
        back without its users row; a flattened chain would return principal 2 or 3
        with NULL users columns, so the Turso compiler must raise. The data is fully
        consistent: principal 2 is a tenant, so it has no users row.
        """
        import sqlite3

        from sqlalchemy import Column, ForeignKey, Integer, MetaData, Table, and_, select
        from sqlalchemy.dialects import sqlite as sqlite_dialect
        from sqlalchemy.exc import CompileError

        metadata = MetaData()
        a = Table(
            "a",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("pid", Integer, ForeignKey("principals.id")),
            Column("uid", Integer, ForeignKey("users.id")),
        )
        principals = Table("principals", metadata, Column("id", Integer, primary_key=True))
        users = Table(
            "users",
            metadata,
            Column("id", Integer, ForeignKey("principals.id"), primary_key=True),
            Column("active", Integer),
        )
        inheritance_on = principals.c.id == users.c.id
        stock_db = sqlite3.connect(":memory:")
        stock_db.executescript(
            "CREATE TABLE a (id INT, pid INT, uid INT);"
            "CREATE TABLE principals (id INT); CREATE TABLE users (id INT, active INT);"
            "INSERT INTO a VALUES (10, 1, 1), (20, 2, NULL), (30, 3, 3);"
            "INSERT INTO principals VALUES (1), (2), (3);"
            "INSERT INTO users VALUES (1, 1), (3, 0);"
        )
        engine = create_async_engine(turso_url(str(tmp_path / "j4.db")), poolclass=NullPool)

        def statement(join):
            columns = (a.c.id, principals.c.id, users.c.id)
            return select(*columns).select_from(join).order_by(a.c.id)

        def stock_sql(stmt):
            literal = {"literal_binds": True}
            return str(stmt.compile(dialect=sqlite_dialect.dialect(), compile_kwargs=literal))

        unsafe = {
            # outer ON names the base table only: principals then users is not total
            "base only": a.outerjoin(
                principals.join(users, inheritance_on), a.c.pid == principals.c.id
            ),
            # outer ON filters the subclass table
            "outer filter": a.outerjoin(
                principals.join(users, inheritance_on),
                and_(a.c.pid == principals.c.id, users.c.active == 1),
            ),
            # the group's own ON filters the subclass table, entered through the base
            "inner filter": a.outerjoin(
                principals.join(users, and_(inheritance_on, users.c.active == 1)),
                a.c.pid == principals.c.id,
            ),
        }
        stock_rows = {
            name: stock_db.execute(stock_sql(statement(join))).fetchall()
            for name, join in unsafe.items()
        }
        # The nested form never reports a principal without its users row.
        assert all(
            row[1] is None or row[2] is not None for rows in stock_rows.values() for row in rows
        )
        for name, join in unsafe.items():
            with pytest.raises(CompileError, match="foreign-key equality"):
                str(statement(join).compile(engine.sync_engine))

        # Safe: the outer ON names the subclass, so users comes first (a filter on
        # it lands on that first step) and principals follows users.id ->
        # principals.id. The flattened SQL returns the nested form's rows.
        safe = {
            "subclass": a.outerjoin(principals.join(users, inheritance_on), a.c.uid == users.c.id),
            "subclass filtered": a.outerjoin(
                principals.join(users, and_(inheritance_on, users.c.active == 1)),
                a.c.uid == users.c.id,
            ),
        }
        for name, join in safe.items():
            flat_sql = str(
                statement(join).compile(engine.sync_engine, compile_kwargs={"literal_binds": True})
            )
            assert "JOIN (" not in flat_sql, flat_sql
            stock = stock_db.execute(stock_sql(statement(join))).fetchall()
            assert stock_db.execute(flat_sql).fetchall() == stock, name
        _run(engine.dispose())

    def test_predicates_never_land_on_another_joins_outer_step(self, tmp_path):
        """A predicate stays on the steps its own join may take, or the rows change.

        Each shape runs flattened and nested on stdlib sqlite3; the rows must match.
        Before the rule, the first two put an inner join's filter onto an earlier
        LEFT JOIN's ON (a row filter became a NULL-ing condition), and the third put
        an outer join's filter onto an earlier inner join's ON (dropping rows).
        """
        import sqlite3

        from sqlalchemy import Column, ForeignKey, Integer, MetaData, Table, and_, select
        from sqlalchemy.dialects import sqlite as sqlite_dialect

        metadata = MetaData()
        a = Table("a", metadata, Column("id", Integer, primary_key=True), Column("flag", Integer))
        b = Table(
            "b",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("aid", Integer),
            Column("x", Integer),
        )
        c = Table("c", metadata, Column("id", Integer, primary_key=True), Column("aid", Integer))
        principals = Table(
            "principals", metadata, Column("id", Integer, primary_key=True), Column("aid", Integer)
        )
        users = Table(
            "users", metadata, Column("id", Integer, ForeignKey("principals.id"), primary_key=True)
        )
        stock_db = sqlite3.connect(":memory:")
        stock_db.executescript(
            "CREATE TABLE a (id INT, flag INT); CREATE TABLE b (id INT, aid INT, x INT);"
            "CREATE TABLE c (id INT, aid INT);"
            "CREATE TABLE principals (id INT, aid INT); CREATE TABLE users (id INT);"
            "INSERT INTO a VALUES (10, 1), (20, 0), (30, 1);"
            "INSERT INTO b VALUES (100, 10, 5); INSERT INTO c VALUES (1, 10), (2, 20);"
            "INSERT INTO principals VALUES (1, 10), (2, 20), (3, 30);"
            "INSERT INTO users VALUES (1), (2), (3);"
        )
        inheritance = principals.join(users, principals.c.id == users.c.id)
        left_outer = a.outerjoin(b, b.c.aid == a.c.id)
        shapes = {
            "inner filter over a left outer join": select(a.c.id, b.c.id, users.c.id)
            .select_from(
                left_outer.join(inheritance, and_(principals.c.aid == a.c.id, a.c.flag == 1))
            )
            .order_by(a.c.id),
            "inner anti-join over a left outer join": select(a.c.id, b.c.id, users.c.id)
            .select_from(
                left_outer.join(inheritance, and_(principals.c.aid == a.c.id, b.c.x.is_(None)))
            )
            .order_by(a.c.id),
            "outer filter over a left inner join": select(a.c.id, c.c.id, users.c.id)
            .select_from(
                a.join(c, c.c.aid == a.c.id).outerjoin(
                    inheritance, and_(users.c.id == c.c.id, a.c.flag == 1)
                )
            )
            .order_by(a.c.id),
        }
        engine = create_async_engine(turso_url(str(tmp_path / "j5.db")), poolclass=NullPool)
        literal = {"literal_binds": True}
        for name, statement in shapes.items():
            flat_sql = str(statement.compile(engine.sync_engine, compile_kwargs=literal))
            stock_sql = str(
                statement.compile(dialect=sqlite_dialect.dialect(), compile_kwargs=literal)
            )
            assert "JOIN (" not in flat_sql, flat_sql
            stock = stock_db.execute(stock_sql).fetchall()
            assert stock_db.execute(flat_sql).fetchall() == stock, (name, flat_sql)
        _run(engine.dispose())

    def test_unverified_join_shapes_fail_to_compile(self, tmp_path):
        """Shapes the flattening has not been verified for raise instead of changing results."""
        from sqlalchemy import Column, Integer, MetaData, Table, select
        from sqlalchemy.exc import CompileError

        metadata = MetaData()
        a = Table("a", metadata, Column("id", Integer), Column("b_id", Integer))
        b = Table("b", metadata, Column("id", Integer), Column("c_id", Integer))
        c = Table("c", metadata, Column("id", Integer))
        engine = create_async_engine(turso_url(str(tmp_path / "j3.db")), poolclass=NullPool)

        # nested OUTER join inside the group
        nested_outer = a.join(b.join(c, b.c.c_id == c.c.id, isouter=True), a.c.b_id == b.c.id)
        with pytest.raises(CompileError, match="nested OUTER join"):
            str(select(a.c.id).select_from(nested_outer).compile(engine.sync_engine))

        # OUTER edge whose group table would get no predicate at all
        unconstrained = a.join(b.join(c, c.c.id == c.c.id), a.c.id == a.c.id, isouter=True)
        with pytest.raises(CompileError, match="no ON predicate"):
            str(select(a.c.id).select_from(unconstrained).compile(engine.sync_engine))

        _run(engine.dispose())

    def test_two_engines_see_each_others_commits(self, tmp_path):
        async def probe():
            url = turso_url(str(tmp_path / "shared.db"))
            first = create_async_engine(url, poolclass=NullPool)
            second = create_async_engine(url, poolclass=NullPool)
            configure_engine(first)
            configure_engine(second)
            async with exclusive_transaction(), first.begin() as connection:
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


# --------------------------------------------------------------------------- #
# A database file open in another process
# --------------------------------------------------------------------------- #
class TestFileInUse:
    LOCK_MESSAGE = (
        "Locking error: Failed locking file '/data/graph.db'. File is locked by another process"
    )

    def test_lock_message_is_recognised_also_when_wrapped(self):
        assert is_locked_by_another_process(RuntimeError(self.LOCK_MESSAGE))
        wrapped = DBAPIError("SELECT 1", (), RuntimeError(self.LOCK_MESSAGE))
        assert is_locked_by_another_process(wrapped)
        assert not is_locked_by_another_process(RuntimeError("database is locked"))
        assert not is_locked_by_another_process(RuntimeError("no such table: t"))

    def test_explain_file_in_use_only_translates_the_lock_error(self):
        from cognee.infrastructure.databases.exceptions import TursoDatabaseInUseError

        with (
            pytest.raises(TursoDatabaseInUseError, match="already open in another process"),
            explain_file_in_use("/data/graph.db"),
        ):
            raise RuntimeError(self.LOCK_MESSAGE)
        with pytest.raises(RuntimeError, match="no such table"), explain_file_in_use("/data/x.db"):
            raise RuntimeError("no such table: t")

    def test_second_process_gets_a_cognee_error(self, tmp_path):
        """A file held by another process fails with TursoDatabaseInUseError, not a driver error.

        Covers both ways cognee opens pyturso: the SQLAlchemy dialect (relational,
        graph, cache engines) and the vector adapter's own connection.
        """
        import subprocess
        import sys

        from cognee.infrastructure.databases.exceptions import TursoDatabaseInUseError
        from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import (
            TursoVectorAdapter,
        )

        engine_file, vector_file = str(tmp_path / "engine.db"), str(tmp_path / "vector.db")
        hold_files = (
            "import sys, time, turso\n"
            "held = [turso.connect(path) for path in sys.argv[1:]]\n"
            "print('held', flush=True)\n"
            "time.sleep(60)"
        )
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                hold_files,
                engine_file,
                vector_file,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout.readline().strip() == "held"

            async def open_engine():
                engine = create_async_engine(turso_url(engine_file), poolclass=NullPool)
                try:
                    async with engine.connect() as connection:
                        await connection.execute(text("SELECT 1"))
                finally:
                    await engine.dispose()

            with pytest.raises(TursoDatabaseInUseError, match="engine.db"):
                _run(open_engine())

            class _Embedding:
                def get_vector_size(self):
                    return 3

            vector = TursoVectorAdapter(
                url=vector_file, api_key=None, embedding_engine=_Embedding()
            )
            with pytest.raises(TursoDatabaseInUseError, match="vector.db"):
                _run(vector.has_collection("Doc_text"))
            _run(vector.close())
        finally:
            holder.kill()
            holder.wait()
