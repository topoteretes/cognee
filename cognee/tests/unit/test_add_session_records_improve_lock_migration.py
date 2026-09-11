import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "b8d3f1a2c4e5_add_session_records_improve_lock.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "session_records_improve_lock_migration", _MIGRATION_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)

LOCK_COLUMNS = {"improve_lock_token", "improve_lock_acquired_at", "improve_rerun_requested"}


def _engine_with_session_records(*, with_lock_columns: bool = False) -> sa.Engine:
    """A real in-memory SQLite DB holding the pre-migration session_records shape."""
    engine = sa.create_engine("sqlite://")
    meta = sa.MetaData()
    columns = [
        sa.Column("session_id", sa.String, primary_key=True),
        sa.Column("user_id", sa.String, nullable=True),
        sa.Column("status", sa.String, nullable=False),
    ]
    if with_lock_columns:
        columns += [
            sa.Column("improve_lock_token", sa.String, nullable=True),
            sa.Column("improve_lock_acquired_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "improve_rerun_requested",
                sa.Boolean,
                nullable=False,
                server_default=sa.false(),
            ),
        ]
    sa.Table("session_records", meta, *columns)
    meta.create_all(engine)
    return engine


def _run(engine: sa.Engine, fn) -> None:
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            fn()


def _columns(engine: sa.Engine) -> set[str]:
    return {col["name"] for col in sa.inspect(engine).get_columns("session_records")}


def test_upgrade_adds_the_lock_columns_with_free_defaults():
    engine = _engine_with_session_records()
    with engine.begin() as conn:
        conn.execute(
            sa.text("INSERT INTO session_records (session_id, status) VALUES ('s1', 'running')")
        )

    _run(engine, migration.upgrade)

    assert LOCK_COLUMNS <= _columns(engine)
    with engine.begin() as conn:
        row = conn.execute(
            sa.text(
                "SELECT improve_lock_token, improve_lock_acquired_at, improve_rerun_requested "
                "FROM session_records"
            )
        ).fetchone()
    # Existing rows come out "free, no rerun pending".
    assert row is not None
    assert row[0] is None and row[1] is None
    assert not row[2]


def test_upgrade_is_idempotent_against_a_real_database():
    engine = _engine_with_session_records()

    for _ in range(2):
        _run(engine, migration.upgrade)

    assert LOCK_COLUMNS <= _columns(engine)


def test_upgrade_is_a_noop_when_create_all_already_added_the_columns():
    engine = _engine_with_session_records(with_lock_columns=True)

    _run(engine, migration.upgrade)

    assert LOCK_COLUMNS <= _columns(engine)


def test_downgrade_removes_the_lock_columns():
    engine = _engine_with_session_records()
    _run(engine, migration.upgrade)

    _run(engine, migration.downgrade)

    assert not (LOCK_COLUMNS & _columns(engine))
    assert {"session_id", "user_id", "status"} <= _columns(engine)


def test_upgrade_and_downgrade_are_noops_without_the_table():
    engine = sa.create_engine("sqlite://")

    _run(engine, migration.upgrade)
    _run(engine, migration.downgrade)

    assert "session_records" not in sa.inspect(engine).get_table_names()
