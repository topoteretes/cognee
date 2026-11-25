"""The capability-layer migration (f6b8d0a2c4e6): what it creates, what it drops, and that
every step is guarded so the chain can be replayed over a database that already has it."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "f6b8d0a2c4e6_add_capability_layer.py"
)
_SPEC = importlib.util.spec_from_file_location("capability_layer_migration", _MIGRATION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)

DROPPED = [name for name, _, _ in migration.DROPPED_TABLES]


def _patch(monkeypatch, tables):
    inspector = MagicMock()
    inspector.get_table_names.return_value = tables
    monkeypatch.setattr(migration.sa, "inspect", lambda _: inspector)
    monkeypatch.setattr(migration.op, "get_bind", lambda: object())
    ops = {
        name: MagicMock() for name in ("create_table", "create_index", "drop_table", "drop_index")
    }
    for name, mock in ops.items():
        monkeypatch.setattr(migration.op, name, mock)
    return ops


def test_upgrade_creates_the_table_with_its_three_foreign_keys_and_drops_the_old_tables(
    monkeypatch,
):
    ops = _patch(monkeypatch, ["principals", "tenants", "users", *DROPPED])

    migration.upgrade()

    ops["create_table"].assert_called_once()
    name, *columns = ops["create_table"].call_args.args
    assert name == migration.TABLE_NAME
    fks = {
        (tuple(c.column_keys), c.ondelete)
        for c in columns
        if isinstance(c, migration.sa.ForeignKeyConstraint)
    }
    # SET NULL on the granter, CASCADE on the holder and the scope.
    assert fks == {
        (("principal_id",), "CASCADE"),
        (("tenant_id",), "CASCADE"),
        (("granted_by",), "SET NULL"),
    }
    assert {c.name for c in columns if isinstance(c, migration.sa.Column)} == {
        "created_at",
        "principal_id",
        "tenant_id",
        "capability",
        "granted_by",
    }
    ops["create_index"].assert_called_once_with(
        migration.INDEX_NAME, migration.TABLE_NAME, ["tenant_id"]
    )
    assert [call.args[0] for call in ops["drop_table"].call_args_list] == DROPPED


def test_upgrade_is_a_no_op_on_a_database_that_already_has_the_layer(monkeypatch):
    ops = _patch(monkeypatch, ["principals", "tenants", "users", migration.TABLE_NAME])

    migration.upgrade()

    ops["create_table"].assert_not_called()
    ops["create_index"].assert_not_called()
    ops["drop_table"].assert_not_called()


def test_downgrade_restores_the_pre_layer_schema(monkeypatch):
    ops = _patch(monkeypatch, ["principals", "tenants", "users", migration.TABLE_NAME])

    migration.downgrade()

    ops["drop_index"].assert_called_once_with(migration.INDEX_NAME, table_name=migration.TABLE_NAME)
    ops["drop_table"].assert_called_once_with(migration.TABLE_NAME)
    assert [call.args[0] for call in ops["create_table"].call_args_list] == DROPPED


def test_downgrade_skips_what_is_already_there(monkeypatch):
    ops = _patch(monkeypatch, ["principals", *DROPPED])

    migration.downgrade()

    ops["drop_table"].assert_not_called()
    ops["create_table"].assert_not_called()
