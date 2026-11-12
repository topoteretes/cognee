"""Revision dd760c09e075 moves pipeline_runs.session_id into pipeline_run_sessions.

SQLite DDL is not transactional, so a process that dies between two statements
leaves the earlier ones in place. Every step of the revision is guarded on its
own, and these tests rerun it from each state an interruption can leave behind.
Real in-memory SQLite, no cognee engine.
"""

import importlib.util
from pathlib import Path
from uuid import uuid4

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "dd760c09e075_add_pipeline_run_sessions.py"
)
_SPEC = importlib.util.spec_from_file_location("pipeline_run_sessions_migration", _MIGRATION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)

RUNS = migration.RUNS_TABLE
SESSIONS = migration.TABLE_NAME
ONE, TWO, NONE = uuid4(), uuid4(), uuid4()


def _engine_before_upgrade() -> sa.Engine:
    """pipeline_runs as the previous head left it: a nullable, indexed session_id."""
    engine = sa.create_engine("sqlite://")
    meta = sa.MetaData()
    sa.Table(
        RUNS,
        meta,
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("pipeline_name", sa.String, nullable=True),
        sa.Column("session_id", sa.String, nullable=True, index=True),
    )
    meta.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                f"INSERT INTO {RUNS} (id, pipeline_name, session_id) VALUES "
                "(:one, 'memify_pipeline', 'chat_1'), (:none, 'add_pipeline', NULL)"
            ),
            {"one": ONE.hex, "none": NONE.hex},
        )
    return engine


def _run(engine: sa.Engine, fn) -> None:
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            fn()


def _columns(engine: sa.Engine, table: str) -> set[str]:
    return {col["name"] for col in sa.inspect(engine).get_columns(table)}


def _indexes(engine: sa.Engine, table: str) -> set[str]:
    return {idx["name"] for idx in sa.inspect(engine).get_indexes(table) if idx["name"]}


def _session_rows(engine: sa.Engine) -> set[tuple[str, str]]:
    with engine.begin() as conn:
        rows = conn.execute(sa.text(f"SELECT pipeline_run_row_id, session_id FROM {SESSIONS}"))
        return {(str(row_id), session_id) for row_id, session_id in rows}


def _column_values(engine: sa.Engine) -> dict[str, str | None]:
    with engine.begin() as conn:
        rows = conn.execute(sa.text(f"SELECT id, session_id FROM {RUNS}"))
        return {str(row_id): session_id for row_id, session_id in rows}


def _add_session_rows(engine: sa.Engine, *pairs: tuple[object, str]) -> None:
    with engine.begin() as conn:
        for row_id, session_id in pairs:
            conn.execute(
                sa.text(
                    f"INSERT INTO {SESSIONS} (pipeline_run_row_id, session_id) VALUES (:r, :s)"
                ),
                {"r": row_id.hex, "s": session_id},
            )


def test_upgrade_copies_the_column_into_the_table_and_drops_it():
    engine = _engine_before_upgrade()

    _run(engine, migration.upgrade)

    assert _session_rows(engine) == {(ONE.hex, "chat_1")}
    assert migration.INDEX_NAME in _indexes(engine, SESSIONS)
    assert "session_id" not in _columns(engine, RUNS)
    assert migration.OLD_INDEX not in _indexes(engine, RUNS)


def test_upgrade_rerun_after_dying_before_the_index_still_creates_it():
    """Table created, process gone before create_index: the rerun must not skip
    the index because the table exists (review finding 2)."""
    engine = _engine_before_upgrade()
    meta = sa.MetaData()
    sa.Table(
        SESSIONS,
        meta,
        sa.Column("pipeline_run_row_id", sa.Uuid, primary_key=True),
        sa.Column("session_id", sa.String, primary_key=True),
    )
    meta.create_all(engine)
    assert _indexes(engine, SESSIONS) == set()

    _run(engine, migration.upgrade)

    assert migration.INDEX_NAME in _indexes(engine, SESSIONS)
    assert _session_rows(engine) == {(ONE.hex, "chat_1")}
    assert "session_id" not in _columns(engine, RUNS)


def test_upgrade_rerun_after_dying_before_the_column_drop_does_not_recopy():
    """Copy committed, process gone before the column drop: the rerun must not
    insert the copied rows again and fail on the primary key."""
    engine = _engine_before_upgrade()
    meta = sa.MetaData()
    sa.Table(
        SESSIONS,
        meta,
        sa.Column("pipeline_run_row_id", sa.Uuid, primary_key=True),
        sa.Column("session_id", sa.String, primary_key=True, index=True),
    )
    meta.create_all(engine)
    _add_session_rows(engine, (ONE, "chat_1"))

    _run(engine, migration.upgrade)

    assert _session_rows(engine) == {(ONE.hex, "chat_1")}
    assert "session_id" not in _columns(engine, RUNS)


def test_downgrade_restores_single_sessions_and_drops_the_table():
    engine = _engine_before_upgrade()
    _run(engine, migration.upgrade)
    with engine.begin() as conn:
        conn.execute(
            sa.text(f"INSERT INTO {RUNS} (id, pipeline_name) VALUES (:two, 'memify_pipeline')"),
            {"two": TWO.hex},
        )
    _add_session_rows(engine, (TWO, "chat_1"), (TWO, "chat_2"))

    _run(engine, migration.downgrade)

    assert _column_values(engine) == {ONE.hex: "chat_1", TWO.hex: None, NONE.hex: None}
    assert migration.OLD_INDEX in _indexes(engine, RUNS)
    assert not sa.inspect(engine).has_table(SESSIONS)


def test_downgrade_rerun_after_dying_before_the_backfill_keeps_the_sessions():
    """Column restored, process gone before the backfill: the rerun must still
    copy the sessions back before it drops their only store (review finding 1)."""
    engine = _engine_before_upgrade()
    _run(engine, migration.upgrade)
    with engine.begin() as conn:
        conn.execute(sa.text(f"ALTER TABLE {RUNS} ADD COLUMN session_id VARCHAR"))
    assert _column_values(engine)[ONE.hex] is None

    _run(engine, migration.downgrade)

    assert _column_values(engine)[ONE.hex] == "chat_1"
    assert migration.OLD_INDEX in _indexes(engine, RUNS)
    assert not sa.inspect(engine).has_table(SESSIONS)


def test_upgrade_and_downgrade_converge_when_run_twice():
    engine = _engine_before_upgrade()
    for _ in range(2):
        _run(engine, migration.upgrade)
    assert _session_rows(engine) == {(ONE.hex, "chat_1")}
    for _ in range(2):
        _run(engine, migration.downgrade)
    assert _column_values(engine)[ONE.hex] == "chat_1"
    assert not sa.inspect(engine).has_table(SESSIONS)
