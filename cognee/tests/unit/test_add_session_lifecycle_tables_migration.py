import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "24f5d4f64d0d_add_session_lifecycle_tables.py"
)
_SPEC = importlib.util.spec_from_file_location("session_lifecycle_migration", _MIGRATION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


def _make_inspector(*, tables=None):
    inspector = MagicMock()
    inspector.get_table_names.return_value = list(tables or [])
    return inspector


def test_upgrade_creates_both_session_tables_and_indexes(monkeypatch):
    inspector = _make_inspector()
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)

    create_table = MagicMock()
    create_index = MagicMock()
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())
    monkeypatch.setattr(migration.op, "create_table", create_table)
    monkeypatch.setattr(migration.op, "create_index", create_index)

    migration.upgrade()

    assert create_table.call_count == 2
    assert create_index.call_count == len(migration.SESSION_RECORDS_INDEXES) + len(
        migration.SESSION_MODEL_USAGE_INDEXES
    )
    create_index.assert_any_call(
        "ix_session_records_user_id", migration.SESSION_RECORDS_TABLE, ["user_id"], unique=False
    )
    create_index.assert_any_call(
        "ix_session_model_usage_user_id",
        migration.SESSION_MODEL_USAGE_TABLE,
        ["user_id"],
        unique=False,
    )


def test_upgrade_skips_existing_session_tables(monkeypatch):
    inspector = _make_inspector(
        tables=[migration.SESSION_RECORDS_TABLE, migration.SESSION_MODEL_USAGE_TABLE]
    )
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)

    create_table = MagicMock()
    create_index = MagicMock()
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())
    monkeypatch.setattr(migration.op, "create_table", create_table)
    monkeypatch.setattr(migration.op, "create_index", create_index)

    migration.upgrade()

    create_table.assert_not_called()
    create_index.assert_not_called()


def test_downgrade_drops_both_session_tables_and_indexes(monkeypatch):
    inspector = _make_inspector(
        tables=[migration.SESSION_RECORDS_TABLE, migration.SESSION_MODEL_USAGE_TABLE]
    )
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)

    drop_index = MagicMock()
    drop_table = MagicMock()
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())
    monkeypatch.setattr(migration.op, "drop_index", drop_index)
    monkeypatch.setattr(migration.op, "drop_table", drop_table)

    migration.downgrade()

    assert drop_index.call_count == len(migration.SESSION_RECORDS_INDEXES) + len(
        migration.SESSION_MODEL_USAGE_INDEXES
    )
    drop_table.assert_any_call(migration.SESSION_MODEL_USAGE_TABLE)
    drop_table.assert_any_call(migration.SESSION_RECORDS_TABLE)


def test_downgrade_skips_missing_session_tables(monkeypatch):
    inspector = _make_inspector()
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)

    drop_index = MagicMock()
    drop_table = MagicMock()
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())
    monkeypatch.setattr(migration.op, "drop_index", drop_index)
    monkeypatch.setattr(migration.op, "drop_table", drop_table)

    migration.downgrade()

    drop_index.assert_not_called()
    drop_table.assert_not_called()
