import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext


_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "b3d5f7a9c1e2_add_graph_metrics_has_full_metrics.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "graph_metrics_has_full_metrics_migration", _MIGRATION_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


def _make_inspector(*, tables=None, columns=None):
    inspector = MagicMock()
    inspector.get_table_names.return_value = tables if tables is not None else ["graph_metrics"]
    inspector.get_columns.return_value = columns or [{"name": "id"}]
    return inspector


def test_upgrade_is_a_noop_when_the_graph_metrics_table_does_not_exist(monkeypatch):
    """A deployment that never ran the GraphMetrics-creating migration (or
    whose relational DB simply has no such table yet) must not crash here."""
    inspector = _make_inspector(tables=[])
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())

    add_column = MagicMock()
    execute = MagicMock()
    monkeypatch.setattr(migration.op, "add_column", add_column)
    monkeypatch.setattr(migration.op, "execute", execute)

    migration.upgrade()

    add_column.assert_not_called()
    execute.assert_not_called()


def test_downgrade_is_a_noop_when_the_graph_metrics_table_does_not_exist(monkeypatch):
    inspector = _make_inspector(tables=[])
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())

    drop_column = MagicMock()
    monkeypatch.setattr(migration.op, "drop_column", drop_column)

    migration.downgrade()

    drop_column.assert_not_called()


def test_upgrade_skips_when_has_full_metrics_column_already_exists(monkeypatch):
    inspector = _make_inspector(columns=[{"name": "id"}, {"name": "has_full_metrics"}])
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())

    add_column = MagicMock()
    execute = MagicMock()
    monkeypatch.setattr(migration.op, "add_column", add_column)
    monkeypatch.setattr(migration.op, "execute", execute)

    migration.upgrade()

    add_column.assert_not_called()
    execute.assert_not_called()


def test_upgrade_adds_the_column_when_the_table_exists_without_it(monkeypatch):
    inspector = _make_inspector()
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())

    add_column = MagicMock()
    execute = MagicMock()
    monkeypatch.setattr(migration.op, "add_column", add_column)
    monkeypatch.setattr(migration.op, "execute", execute)

    migration.upgrade()

    add_column.assert_called_once()
    execute.assert_called_once()


def _run_upgrade_against_real_sqlite(rows):
    """Exercise upgrade() against a real in-memory SQLite DB (not mocks), so
    the backfill's actual SQL — not a mocked call — is what gets checked."""
    engine = sa.create_engine("sqlite://")
    meta = sa.MetaData()
    graph_metrics = sa.Table(
        "graph_metrics",
        meta,
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("num_tokens", sa.Integer, nullable=True),
        sa.Column("diameter", sa.Integer, nullable=True),
    )
    meta.create_all(engine)

    with engine.begin() as conn:
        conn.execute(graph_metrics.insert(), rows)

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            migration.upgrade()

    with engine.begin() as conn:
        result = conn.execute(sa.text("SELECT id, has_full_metrics FROM graph_metrics"))
        return dict(result.fetchall())


def test_backfill_marks_a_full_metrics_row_even_when_its_token_sum_is_null():
    """The bug this migration used to have: num_tokens is a SQL SUM() that
    legitimately comes back NULL for a fully-computed run over a dataset with
    no token_count data. diameter is always set on the full path (a real
    value or -1), so it — not num_tokens — is what backfill keys on."""
    has_full_metrics = _run_upgrade_against_real_sqlite(
        [{"id": "full-null-tokens", "num_tokens": None, "diameter": -1}]
    )

    assert has_full_metrics["full-null-tokens"] == 1


def test_backfill_marks_a_partial_counting_only_row_as_not_full():
    has_full_metrics = _run_upgrade_against_real_sqlite(
        [{"id": "partial-row", "num_tokens": None, "diameter": None}]
    )

    assert has_full_metrics["partial-row"] == 0


def test_backfill_marks_an_ordinary_full_metrics_row_as_full():
    has_full_metrics = _run_upgrade_against_real_sqlite(
        [{"id": "full-with-tokens", "num_tokens": 42, "diameter": 3}]
    )

    assert has_full_metrics["full-with-tokens"] == 1


def test_upgrade_is_idempotent_against_a_real_database():
    """Running upgrade() twice must not fail on "column already exists"."""
    engine = sa.create_engine("sqlite://")
    meta = sa.MetaData()
    graph_metrics = sa.Table(
        "graph_metrics",
        meta,
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("num_tokens", sa.Integer, nullable=True),
        sa.Column("diameter", sa.Integer, nullable=True),
    )
    meta.create_all(engine)

    with engine.begin() as conn:
        conn.execute(graph_metrics.insert(), [{"id": "row", "num_tokens": 1, "diameter": 1}])

    for _ in range(2):
        with engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with Operations.context(ctx):
                migration.upgrade()

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            migration.downgrade()
