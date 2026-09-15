import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "b2d6f4a8c150_add_pipeline_runs_last_heartbeat_at.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "pipeline_runs_last_heartbeat_at_migration", _MIGRATION_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


def _engine_with_pipeline_runs(*, with_heartbeat: bool = False) -> sa.Engine:
    """A real in-memory SQLite DB holding the pre-migration pipeline_runs shape."""
    engine = sa.create_engine("sqlite://")
    meta = sa.MetaData()
    columns = [
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String, nullable=True),
        sa.Column("pipeline_run_id", sa.String, nullable=True),
    ]
    if with_heartbeat:
        columns.append(sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    sa.Table("pipeline_runs", meta, *columns)
    meta.create_all(engine)
    return engine


def _run(engine: sa.Engine, fn) -> None:
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            fn()


def _columns(engine: sa.Engine) -> set[str]:
    return {col["name"] for col in sa.inspect(engine).get_columns("pipeline_runs")}


def test_upgrade_adds_the_nullable_column_and_leaves_existing_rows_null():
    """No backfill by design: an existing row has produced no evidence of
    liveness, and stamping created_at into the column would fake one."""
    engine = _engine_with_pipeline_runs()
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO pipeline_runs (id, created_at, status) "
                "VALUES ('r1', '2026-01-01 00:00:00', 'DATASET_PROCESSING_STARTED')"
            )
        )

    _run(engine, migration.upgrade)

    assert "last_heartbeat_at" in _columns(engine)
    with engine.begin() as conn:
        row = conn.execute(sa.text("SELECT last_heartbeat_at FROM pipeline_runs")).fetchone()
    assert row is not None and row[0] is None


def test_upgrade_is_idempotent_against_a_real_database():
    """Running upgrade() twice must not fail on 'column already exists'."""
    engine = _engine_with_pipeline_runs()

    for _ in range(2):
        _run(engine, migration.upgrade)

    assert "last_heartbeat_at" in _columns(engine)


def test_upgrade_is_a_noop_when_create_all_already_added_the_column():
    """A fresh database is built by create_all() from the current models and
    then stamped at head, so the column is already there when the chain runs."""
    engine = _engine_with_pipeline_runs(with_heartbeat=True)

    _run(engine, migration.upgrade)

    assert "last_heartbeat_at" in _columns(engine)


def test_downgrade_removes_the_column():
    engine = _engine_with_pipeline_runs()
    _run(engine, migration.upgrade)

    _run(engine, migration.downgrade)

    assert "last_heartbeat_at" not in _columns(engine)


def test_upgrade_and_downgrade_are_noops_without_the_table():
    engine = sa.create_engine("sqlite://")

    _run(engine, migration.upgrade)
    _run(engine, migration.downgrade)

    assert "pipeline_runs" not in sa.inspect(engine).get_table_names()
