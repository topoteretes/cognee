"""The migration creates the table the model declares, and can run twice or backwards (SDK-1434).

The lockstep guard compares tables and columns; this also compares indexes (names,
columns, uniqueness, partial conditions) and foreign keys, since the due scan and
the one-default-row rule depend on them.
"""

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

import cognee

# Registering these tables is what lets the policy table's foreign keys resolve.
import cognee.modules.integrations.models
import cognee.modules.users.models
from cognee.modules.integrations.sync_policy.models import IntegrationSyncPolicy

TABLE = "integration_sync_policies"
MIGRATION = (
    Path(cognee.__file__).parent / "alembic/versions/91e29f0d0467_add_integration_sync_policies.py"
)


def _migration():
    spec = importlib.util.spec_from_file_location("sync_policies_migration", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(engine, step):
    with engine.begin() as connection, Operations.context(MigrationContext.configure(connection)):
        step()


def _shape(engine):
    inspector = sa.inspect(engine)
    columns = sorted(
        (c["name"], str(c["type"]), c["nullable"], str(c["default"]))
        for c in inspector.get_columns(TABLE)
    )
    indexes = sorted(
        (
            index["name"],
            tuple(index["column_names"]),
            bool(index["unique"]),
            str(index.get("dialect_options", {}).get("sqlite_where")),
        )
        for index in inspector.get_indexes(TABLE)
    )
    foreign_keys = sorted(
        (tuple(fk["constrained_columns"]), fk["referred_table"], fk["options"].get("ondelete"))
        for fk in inspector.get_foreign_keys(TABLE)
    )
    return columns, indexes, foreign_keys


def test_the_migration_creates_what_the_model_declares(tmp_path):
    from_model = sa.create_engine(f"sqlite:///{tmp_path}/model.db")
    from_migration = sa.create_engine(f"sqlite:///{tmp_path}/migration.db")
    IntegrationSyncPolicy.__table__.create(from_model)

    _run(from_migration, _migration().upgrade)

    assert _shape(from_migration) == _shape(from_model)


def test_the_migration_finishes_a_half_applied_upgrade_and_reverses(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path}/migration.db")
    migration = _migration()
    _run(engine, migration.upgrade)
    complete = _shape(engine)
    with engine.begin() as connection:
        connection.execute(sa.text("DROP INDEX ix_integration_sync_policies_due"))
    assert _shape(engine) != complete

    _run(engine, migration.upgrade)
    assert _shape(engine) == complete

    _run(engine, migration.downgrade)
    assert not sa.inspect(engine).has_table(TABLE)
    # Reversing a database that is already reversed is a no-op.
    _run(engine, migration.downgrade)
