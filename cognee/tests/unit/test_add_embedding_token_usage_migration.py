"""Tests for migration d4a8e2f6b1c9 (embedding token usage columns, SDK-1077).

Run against a real in-memory sqlite database rather than a mocked ``op``, so
they prove what matters: the columns land, existing session rows read as
zero embedding usage and existing per-model rows as LLM rows, a second
upgrade is a no-op, downgrade drops only these columns, and a database
missing one of the tables is left alone.
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
    / "d4a8e2f6b1c9_add_embedding_token_usage.py"
)
_SPEC = importlib.util.spec_from_file_location("embedding_token_usage_migration", _MIGRATION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


def _pre_migration_metadata(omit_table: str = "") -> sa.MetaData:
    """The three tables as they stand BEFORE this migration, trimmed to a key and
    an existing token column. Not the live models: those already declare the
    new columns. ``omit_table`` leaves one table out."""
    metadata = sa.MetaData()
    tables = {
        "pipeline_runs": [sa.Column("id", sa.String(), primary_key=True)],
        "session_records": [sa.Column("session_id", sa.String(), primary_key=True)],
        "session_model_usage": [
            sa.Column("session_id", sa.String(), primary_key=True),
            sa.Column("model", sa.Text(), primary_key=True),
        ],
    }
    for name, columns in tables.items():
        if name != omit_table:
            sa.Table(name, metadata, *columns, sa.Column("tokens_in", sa.Integer()))
    return metadata


@contextmanager
def _database(omit_table: str = ""):
    """Scratch sqlite database with the pre-migration tables, one existing row each."""
    with sa.create_engine("sqlite://").connect() as connection:
        metadata = _pre_migration_metadata(omit_table)
        metadata.create_all(connection)
        for table in metadata.sorted_tables:
            key = {column.name: "existing" for column in table.primary_key.columns}
            connection.execute(table.insert().values(**key, tokens_in=5))
        yield connection


def _run(connection, step) -> None:
    """Invoke a migration function with alembic's ``op`` proxy bound to conn."""
    with Operations.context(MigrationContext.configure(connection)):
        step()


def _column_names(connection, table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(connection).get_columns(table)}


def test_upgrade_adds_the_columns_and_backfills_existing_rows():
    with _database() as conn:
        _run(conn, migration.upgrade)

        runs = conn.execute(sa.text("SELECT embedding_tokens FROM pipeline_runs")).one()
        sessions = conn.execute(
            sa.text("SELECT embedding_tokens, embedding_cost_usd FROM session_records")
        ).one()
        models = conn.execute(sa.text("SELECT kind FROM session_model_usage")).one()

    assert runs == (None,)  # not measured
    assert sessions == (0, 0.0)
    assert models == ("llm",)


def test_a_second_upgrade_is_a_no_op():
    with _database() as conn:
        _run(conn, migration.upgrade)
        _run(conn, migration.upgrade)

        assert "kind" in _column_names(conn, "session_model_usage")
        assert {"embedding_tokens", "embedding_cost_usd"} <= _column_names(conn, "session_records")


def test_downgrade_drops_only_these_columns():
    with _database() as conn:
        _run(conn, migration.upgrade)
        _run(conn, migration.downgrade)

        assert _column_names(conn, "pipeline_runs") == {"id", "tokens_in"}
        assert _column_names(conn, "session_records") == {"session_id", "tokens_in"}
        assert _column_names(conn, "session_model_usage") == {"session_id", "model", "tokens_in"}


@pytest.mark.parametrize("steps", [["upgrade"], ["upgrade", "downgrade"]])
def test_a_missing_table_is_left_alone(steps):
    with _database(omit_table="session_model_usage") as conn:
        for step in steps:
            _run(conn, getattr(migration, step))

        assert "session_model_usage" not in sa.inspect(conn).get_table_names()
        upgraded = steps == ["upgrade"]
        assert ("embedding_tokens" in _column_names(conn, "session_records")) == upgraded
