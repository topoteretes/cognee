import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "7c5d4e2f8a91_add_parent_user_id_to_users.py"
)
_SPEC = importlib.util.spec_from_file_location("parent_user_id_migration", _MIGRATION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


def _make_inspector(*, columns=None, foreign_keys=None, tables=None):
    inspector = MagicMock()
    inspector.get_table_names.return_value = tables or [migration.TABLE_NAME]
    inspector.get_columns.return_value = columns or [{"name": "id"}]
    inspector.get_foreign_keys.return_value = foreign_keys or []
    return inspector


def _make_context(dialect_name: str):
    return SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))


def test_upgrade_adds_parent_user_id_column_and_fk(monkeypatch):
    inspector = _make_inspector()
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)

    add_column = MagicMock()
    create_foreign_key = MagicMock()
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())
    monkeypatch.setattr(migration.op, "get_context", lambda: _make_context("postgresql"))
    monkeypatch.setattr(migration.op, "add_column", add_column)
    monkeypatch.setattr(migration.op, "create_foreign_key", create_foreign_key)

    migration.upgrade()

    add_column.assert_called_once()
    create_foreign_key.assert_called_once_with(
        migration.FK_NAME,
        migration.TABLE_NAME,
        migration.TABLE_NAME,
        [migration.COLUMN_NAME],
        ["id"],
        ondelete="SET NULL",
    )


def test_upgrade_skips_when_parent_user_id_column_exists(monkeypatch):
    inspector = _make_inspector(
        columns=[{"name": "id"}, {"name": migration.COLUMN_NAME}],
        foreign_keys=[{"name": migration.FK_NAME}],
    )
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)

    add_column = MagicMock()
    create_foreign_key = MagicMock()
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())
    monkeypatch.setattr(migration.op, "get_context", lambda: _make_context("postgresql"))
    monkeypatch.setattr(migration.op, "add_column", add_column)
    monkeypatch.setattr(migration.op, "create_foreign_key", create_foreign_key)

    migration.upgrade()

    add_column.assert_not_called()
    create_foreign_key.assert_not_called()
