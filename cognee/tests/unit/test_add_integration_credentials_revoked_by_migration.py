import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "e3f1a9c5b7d2_add_integration_credentials_revoked_by.py"
)
_SPEC = importlib.util.spec_from_file_location("revoked_by_migration", _MIGRATION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


def _engine_with_credentials() -> sa.Engine:
    """A real in-memory SQLite DB holding the pre-migration integration_credentials shape."""
    engine = sa.create_engine("sqlite://")
    meta = sa.MetaData()
    sa.Table(
        "integration_credentials",
        meta,
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("revoked_at", sa.DateTime, nullable=True),
    )
    meta.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            sa.text("INSERT INTO integration_credentials (id, status) VALUES ('c1', 'revoked')")
        )
    return engine


def _run(engine: sa.Engine, fn) -> None:
    with engine.begin() as conn, Operations.context(MigrationContext.configure(conn)):
        fn()


def _columns(engine: sa.Engine) -> set[str]:
    return {col["name"] for col in sa.inspect(engine).get_columns("integration_credentials")}


def test_upgrade_adds_revoked_by_and_leaves_existing_rows_unknown():
    engine = _engine_with_credentials()

    _run(engine, migration.upgrade)

    assert "revoked_by" in _columns(engine)
    with engine.begin() as conn:
        row = conn.execute(sa.text("SELECT revoked_by FROM integration_credentials")).fetchone()
    assert row is not None and row[0] is None


def test_upgrade_and_downgrade_are_idempotent():
    engine = _engine_with_credentials()

    for _ in range(2):
        _run(engine, migration.upgrade)
    assert "revoked_by" in _columns(engine)

    for _ in range(2):
        _run(engine, migration.downgrade)
    assert "revoked_by" not in _columns(engine)


def test_a_database_without_the_table_is_left_alone():
    engine = sa.create_engine("sqlite://")

    _run(engine, migration.upgrade)
    _run(engine, migration.downgrade)

    assert not sa.inspect(engine).has_table("integration_credentials")
