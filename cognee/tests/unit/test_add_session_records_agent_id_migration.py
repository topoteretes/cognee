import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext


_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "a7c2e9f4b8d1_add_session_records_agent_id.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "session_records_agent_id_migration", _MIGRATION_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


def _engine_with_session_records(*, with_agent_id: bool = False) -> sa.Engine:
    """A real in-memory SQLite DB holding the pre-migration session_records shape."""
    engine = sa.create_engine("sqlite://")
    meta = sa.MetaData()
    columns = [
        sa.Column("session_id", sa.String, primary_key=True),
        sa.Column("user_id", sa.String, nullable=True),
        sa.Column("status", sa.String, nullable=False),
    ]
    if with_agent_id:
        columns.append(sa.Column("agent_id", sa.String, nullable=True))
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


def _indexes(engine: sa.Engine) -> set[str]:
    return {
        idx["name"]
        for idx in sa.inspect(engine).get_indexes("session_records")
        if idx["name"] is not None
    }


def test_upgrade_adds_nullable_agent_id_column_and_index():
    engine = _engine_with_session_records()
    with engine.begin() as conn:
        conn.execute(
            sa.text("INSERT INTO session_records (session_id, status) VALUES ('s1', 'active')")
        )

    _run(engine, migration.upgrade)

    assert "agent_id" in _columns(engine)
    assert "ix_session_records_agent_id" in _indexes(engine)
    with engine.begin() as conn:
        # Existing rows survive with NULL agent_id; the lifecycle code fills it later.
        row = conn.execute(sa.text("SELECT agent_id FROM session_records")).fetchone()
    assert row is not None and row[0] is None


def test_upgrade_is_idempotent_against_a_real_database():
    """Running upgrade() twice must not fail on 'column already exists'."""
    engine = _engine_with_session_records()

    for _ in range(2):
        _run(engine, migration.upgrade)

    assert "agent_id" in _columns(engine)
    assert "ix_session_records_agent_id" in _indexes(engine)


def test_upgrade_adds_only_the_index_when_create_all_already_added_the_column():
    """A DB whose table came from metadata.create_all() on the new model already
    has the column but not necessarily the index; upgrade fills the gap."""
    engine = _engine_with_session_records(with_agent_id=True)

    _run(engine, migration.upgrade)

    assert "agent_id" in _columns(engine)
    assert "ix_session_records_agent_id" in _indexes(engine)


def test_downgrade_removes_index_and_column():
    engine = _engine_with_session_records()
    _run(engine, migration.upgrade)

    _run(engine, migration.downgrade)

    assert "agent_id" not in _columns(engine)
    assert "ix_session_records_agent_id" not in _indexes(engine)


def test_upgrade_and_downgrade_are_noops_without_the_table():
    """A deployment that never ran the session-lifecycle migration must not crash here."""
    engine = sa.create_engine("sqlite://")

    _run(engine, migration.upgrade)
    _run(engine, migration.downgrade)

    assert "session_records" not in sa.inspect(engine).get_table_names()
