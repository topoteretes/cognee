import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "b8d0f2a4c6e8_add_granted_by_to_principal_capabilities.py"
)
_SPEC = importlib.util.spec_from_file_location("granted_by_migration", _MIGRATION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


def _make_inspector(*, columns=None, foreign_keys=None, tables=None):
    inspector = MagicMock()
    inspector.get_table_names.return_value = [migration.TABLE_NAME] if tables is None else tables
    inspector.get_columns.return_value = columns or [{"name": "principal_id"}]
    inspector.get_foreign_keys.return_value = foreign_keys or []
    return inspector


def _make_context(dialect_name: str):
    return SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))


def _patch_op(monkeypatch, dialect_name):
    add_column = MagicMock()
    create_foreign_key = MagicMock()
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())
    monkeypatch.setattr(migration.op, "get_context", lambda: _make_context(dialect_name))
    monkeypatch.setattr(migration.op, "add_column", add_column)
    monkeypatch.setattr(migration.op, "create_foreign_key", create_foreign_key)
    return add_column, create_foreign_key


def test_upgrade_adds_granted_by_column_and_a_set_null_fk_to_users(monkeypatch):
    """SET NULL, not CASCADE: removing the granter must not revoke what they granted."""
    inspector = _make_inspector()
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)
    add_column, create_foreign_key = _patch_op(monkeypatch, "postgresql")

    migration.upgrade()

    add_column.assert_called_once()
    create_foreign_key.assert_called_once_with(
        migration.FK_NAME,
        migration.TABLE_NAME,
        "users",
        [migration.COLUMN_NAME],
        ["id"],
        ondelete="SET NULL",
    )


def test_upgrade_skips_when_granted_by_already_exists(monkeypatch):
    inspector = _make_inspector(
        columns=[{"name": "principal_id"}, {"name": migration.COLUMN_NAME}],
        foreign_keys=[{"name": migration.FK_NAME}],
    )
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)
    add_column, create_foreign_key = _patch_op(monkeypatch, "postgresql")

    migration.upgrade()

    add_column.assert_not_called()
    create_foreign_key.assert_not_called()


def test_upgrade_skips_when_the_table_does_not_exist(monkeypatch):
    inspector = _make_inspector(tables=[])
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)
    add_column, create_foreign_key = _patch_op(monkeypatch, "postgresql")

    migration.upgrade()

    add_column.assert_not_called()
    create_foreign_key.assert_not_called()
