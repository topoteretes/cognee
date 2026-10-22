"""Tests for migration b7e4c2a9d6f1 (drop the data key from pipeline_runs.run_info).

Run against a real in-memory sqlite database rather than a mocked ``op``, so
they prove what matters about a data migration: every row holding ``data`` or
``errored_data`` loses exactly those keys, whatever they held (input text, an
id list, the literal "None"), every other key of those rows survives, rows
without the keys are untouched, no text or id list remains in the column, a
second upgrade is a no-op, a database without the table is a no-op, and the
Python scan used on dialects without a JSON key test selects the same rows as
the sqlite predicate.
"""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "b7e4c2a9d6f1_drop_run_info_data.py"
)
_SPEC = importlib.util.spec_from_file_location("drop_run_info_data_migration", _MIGRATION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)

TEXT_PREVIEW = "The door code for my apartment building is 8919-6729. Please remember it."
TRUNCATED = "x" * 512 + "... [truncated, 9000 chars total]"
ID_LIST = [str(uuid4()), str(uuid4())]

# What earlier releases could have stored, keyed by what the row stands for.
SEED = {
    "add_text": {"data": TEXT_PREVIEW, "recovery_lock": "lock-1"},
    "add_text_truncated": {"data": TRUNCATED, "error": "boom"},
    "empty_input": {"data": "None"},
    "cognify_ids": {"data": ID_LIST, "progress": {"completed_items": 2, "total_items": 2}},
    "prerelease_ids": {"data": [ID_LIST[0]], "errored_data": [ID_LIST[1]], "error": "boom"},
    "no_data_key": {"error": "AbandonedPipelineRunError"},
    "improve_stamp": {"triplet_enrichment_started_at": "2026-10-01T00:00:00+00:00"},
    "null_run_info": None,
}
AFFECTED = {"add_text", "add_text_truncated", "empty_input", "cognify_ids", "prerelease_ids"}
EXPECTED_AFTER = {
    "add_text": {"recovery_lock": "lock-1"},
    "add_text_truncated": {"error": "boom"},
    "empty_input": {},
    "cognify_ids": {"progress": {"completed_items": 2, "total_items": 2}},
    "prerelease_ids": {"error": "boom"},
}


def _pre_migration_table() -> sa.Table:
    return sa.Table(
        migration.TABLE_NAME,
        sa.MetaData(),
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_info", sa.JSON(), nullable=True),
    )


@contextmanager
def _database(create_table: bool = True):
    with sa.create_engine("sqlite://").connect() as connection:
        if create_table:
            _pre_migration_table().create(connection)
        yield connection


def _seed(connection) -> dict[str, object]:
    table = _pre_migration_table()
    ids = {name: uuid4() for name in SEED}
    connection.execute(
        sa.insert(table),
        [{"id": ids[name], "run_info": run_info} for name, run_info in SEED.items()],
    )
    return ids


def _run(connection, step) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        step()


def _stored(connection, ids) -> dict[str, object]:
    table = _pre_migration_table()
    rows = connection.execute(sa.select(table.c.id, table.c.run_info)).all()
    by_id = {row_id: run_info for row_id, run_info in rows}
    return {name: by_id[row_id] for name, row_id in ids.items()}


@pytest.fixture
def seeded():
    with _database() as connection:
        ids = _seed(connection)
        yield connection, ids


def test_upgrade_drops_the_keys_and_keeps_the_rest(seeded):
    connection, ids = seeded

    _run(connection, migration.upgrade)

    stored = _stored(connection, ids)
    for name, expected in EXPECTED_AFTER.items():
        assert stored[name] == expected, name
    for name in SEED.keys() - AFFECTED:
        assert stored[name] == SEED[name], name


def test_no_text_or_id_list_survives_anywhere_in_the_column(seeded):
    connection, _ids = seeded

    _run(connection, migration.upgrade)

    table = _pre_migration_table()
    column_text = " ".join(
        str(run_info) for (run_info,) in connection.execute(sa.select(table.c.run_info))
    )
    assert "8919-6729" not in column_text
    assert "truncated" not in column_text
    for data_id in ID_LIST:
        assert data_id not in column_text


def test_upgrade_is_idempotent(seeded):
    connection, ids = seeded
    _run(connection, migration.upgrade)
    after_first = _stored(connection, ids)

    _run(connection, migration.upgrade)

    assert _stored(connection, ids) == after_first


def test_upgrade_without_the_table_is_a_noop():
    with _database(create_table=False) as connection:
        _run(connection, migration.upgrade)
        assert migration.TABLE_NAME not in sa.inspect(connection).get_table_names()


def test_downgrade_restores_nothing(seeded):
    connection, ids = seeded
    _run(connection, migration.upgrade)
    after_upgrade = _stored(connection, ids)

    _run(connection, migration.downgrade)

    assert _stored(connection, ids) == after_upgrade


def test_python_scan_selects_the_same_rows_as_the_sqlite_predicate(seeded, monkeypatch):
    """A dialect without a JSON key test reads every row; the result must match."""
    connection, ids = seeded
    table = migration._table()
    filtered = migration.rows_with_dropped_keys(connection, table)

    monkeypatch.setattr(migration, "dropped_key_filter", lambda dialect_name: None)
    scanned = migration.rows_with_dropped_keys(connection, table)

    assert sorted(scanned, key=repr) == sorted(filtered, key=repr)
    assert {row_id for row_id, _ in scanned} == {ids[name] for name in AFFECTED}


@pytest.mark.parametrize(
    "dialect_name,has_predicate",
    [("sqlite", True), ("postgresql", True), ("mysql", False), ("mssql", False)],
)
def test_predicate_exists_only_for_dialects_with_a_json_key_test(dialect_name, has_predicate):
    assert (migration.dropped_key_filter(dialect_name) is not None) is has_predicate


def test_rows_beyond_one_batch_are_all_stripped():
    with _database() as connection:
        table = _pre_migration_table()
        ids = [uuid4() for _ in range(migration.BATCH_SIZE * 2 + 7)]
        connection.execute(
            sa.insert(table),
            [{"id": row_id, "run_info": {"data": [str(row_id)], "x": 1}} for row_id in ids],
        )

        _run(connection, migration.upgrade)

        remaining = connection.execute(
            sa.select(sa.func.count())
            .select_from(table)
            .where(sa.text("json_type(run_info, '$.data') IS NOT NULL"))
        ).scalar_one()
        assert remaining == 0
        kept = connection.execute(sa.select(table.c.run_info)).first()[0]
        assert kept == {"x": 1}
