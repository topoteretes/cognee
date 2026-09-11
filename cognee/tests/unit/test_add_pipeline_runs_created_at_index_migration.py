"""Tests for migration c4e8a1f6b3d7 (index pipeline_runs on created_at, id).

Run against a real in-memory sqlite database rather than a mocked ``op``, so
they prove what actually matters about an index migration: that the index
lands, that a second upgrade is a no-op (the migration runs against
already-migrated databases), that downgrade drops only this index and leaves
SDK-399's alone, and that a pipeline_runs missing an indexed column fails
loudly instead of stamping a phantom index.
"""

import importlib.util
from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "c4e8a1f6b3d7_add_pipeline_runs_created_at_index.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "pipeline_runs_created_at_index_migration", _MIGRATION_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)

# An SDK-399 index that must survive this migration's downgrade.
SDK_399_INDEX = "ix_pipeline_runs_user_id"


def _pre_migration_table(omit: str = "") -> sa.Table:
    """pipeline_runs as it stands BEFORE this migration: created_at unindexed.

    Deliberately not PipelineRun.__table__ — that now declares the composite
    index itself, so create_all() would leave nothing for upgrade() to do.
    ``omit`` drops one indexed column, standing in for a pipeline_runs that
    is not the table cognee defines.
    """
    columns = [
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("user_id", sa.UUID(), index=True),
    ]
    return sa.Table(
        migration.TABLE_NAME,
        sa.MetaData(),
        *[column for column in columns if column.name != omit],
    )


@contextmanager
def _database(omit: str = ""):
    """Scratch sqlite database holding a pre-migration pipeline_runs."""
    with sa.create_engine("sqlite://").connect() as connection:
        _pre_migration_table(omit).create(connection)
        yield connection


def _run(connection, step) -> None:
    """Invoke a migration function with alembic's ``op`` proxy bound to conn."""
    with Operations.context(MigrationContext.configure(connection)):
        step()


def _index_columns(connection, index_name: str):
    for index in sa.inspect(connection).get_indexes(migration.TABLE_NAME):
        if index["name"] == index_name:
            return index["column_names"]
    return None


@pytest.fixture
def conn():
    with _database() as connection:
        yield connection


def test_upgrade_creates_composite_index_on_created_at_and_id(conn):
    assert _index_columns(conn, migration.INDEX_NAME) is None

    _run(conn, migration.upgrade)

    assert _index_columns(conn, migration.INDEX_NAME) == migration.INDEX_COLUMNS


def test_upgrade_is_idempotent(conn):
    _run(conn, migration.upgrade)
    _run(conn, migration.upgrade)

    assert _index_columns(conn, migration.INDEX_NAME) == migration.INDEX_COLUMNS


def test_downgrade_drops_only_this_index(conn):
    _run(conn, migration.upgrade)

    _run(conn, migration.downgrade)

    assert _index_columns(conn, migration.INDEX_NAME) is None
    assert _index_columns(conn, SDK_399_INDEX) == ["user_id"]


def test_downgrade_is_idempotent(conn):
    _run(conn, migration.downgrade)

    assert _index_columns(conn, migration.INDEX_NAME) is None


def test_migration_skips_when_table_is_absent(conn):
    _pre_migration_table().drop(conn)

    _run(conn, migration.upgrade)
    _run(conn, migration.downgrade)

    assert migration.TABLE_NAME not in sa.inspect(conn).get_table_names()


@pytest.mark.parametrize("missing", migration.INDEX_COLUMNS)
def test_upgrade_raises_when_an_indexed_column_is_missing(missing):
    """Skipping would let alembic stamp this revision over a database that
    never got the index, hiding the slow scans it exists to remove."""
    with _database(omit=missing) as conn:
        with pytest.raises(RuntimeError, match=missing):
            _run(conn, migration.upgrade)

        assert _index_columns(conn, migration.INDEX_NAME) is None


@pytest.mark.parametrize("missing", migration.INDEX_COLUMNS)
def test_downgrade_still_succeeds_when_an_indexed_column_is_missing(missing):
    """Rollback must not be held hostage by a schema this revision never
    touched — the index cannot exist without the columns anyway."""
    with _database(omit=missing) as conn:
        _run(conn, migration.downgrade)

        assert _index_columns(conn, migration.INDEX_NAME) is None
