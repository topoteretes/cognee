"""Integration tests for the reconcile revision (``1c22e6cb5aec``) on real SQLite.

The reconcile revision is the chain's last migration: it converges every
existing database — whatever model surface the process that CREATED it
happened to import — to the chain's frozen truth, additively. These tests pin
its contract:

1. a chain-born database contains everything already and passes through as a
   no-op — also for a second database migrated in the same process (the
   frozen MetaData is defined once per process);
2. a legacy database bootstrapped from a PARTIAL model surface (the pure-SDK
   1.5.x shape: ``create_all`` without the router-only tables, stamped at the
   pre-reconcile head) converges: the missing tables appear, with their
   indexes, and the result matches a chain-born database name for name;
3. a column missing from an existing table returns with its server default —
   including the ``now()`` default that SQLite's ADD COLUMN rejects (the
   batch-mode table rebuild);
4. a nullable foreign-key column returns with its constraint;
5. a NOT NULL column without a server default, missing from a POPULATED table,
   aborts the migration before any change.
"""

import asyncio
import importlib
import importlib.util
import tempfile
import uuid
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

# The revision just below the reconcile: stamping a damaged database here makes
# an upgrade run ONLY the reconcile, isolating its behaviour.
PRE_RECONCILE_HEAD = "f3a7b9c1d2e4"
# The freeze point: the head at which the initial revision's frozen surface was
# certified. A legacy database born by create_all under the old bootstrap holds
# (a subset of) THAT surface and is stamped there or below; on upgrade it
# replays everything above it — the post-freeze revisions and the reconcile.
FREEZE_HEAD = "d1e2f3a4b5c6"
# Registered only by the API routers, never by ``import cognee``: the tables a
# pure-SDK process never created under the create_all bootstrap.
ROUTER_ONLY_TABLES = {"integration_credentials", "sync_operations"}


def _register_all_models() -> None:
    import cognee.modules

    root = Path(cognee.modules.__file__).parent
    for sub in sorted(path.name for path in root.iterdir() if path.is_dir()):
        if (root / sub / "models" / "__init__.py").exists():
            importlib.import_module(f"cognee.modules.{sub}.models")


def _frozen_table_names() -> set[str]:
    """The tables the initial revision's frozen surface carries, read from the
    reconcile revision's own declaration (loaded by path; a migration module is
    not importable by name). Needs an Alembic op context for its enum helper."""
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext
    from cognee.modules.migrations.lockstep import packaged_script_location

    path = (
        Path(packaged_script_location()) / "versions" / "1c22e6cb5aec_reconcile_to_frozen_schema.py"
    )
    spec = importlib.util.spec_from_file_location("reconcile_revision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with sa.create_engine("sqlite://").connect() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            module._define_frozen_schema()
    return set(module._FROZEN.tables)


def _adapter():
    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )

    return SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tempfile.mkdtemp()}/db.sqlite")


def _chain_head() -> str:
    from cognee.modules.migrations.startup import _build_alembic_config

    return ScriptDirectory.from_config(_build_alembic_config()).get_current_head()


async def _upgrade(adapter) -> None:
    from cognee.modules.migrations.startup import run_relational_migrations

    await run_relational_migrations("head", engine=adapter)


async def _snapshot(adapter) -> dict:
    """Name-level schema: per table its columns (type, nullability, default),
    its index names with columns, and its foreign keys as (columns -> target)
    — constraint NAMES are left out on purpose (create_all-born and chain-born
    databases name them differently; the constraints are the same)."""

    def reflect(sync_conn):
        inspector = sa.inspect(sync_conn)
        snap = {}
        for table in inspector.get_table_names():
            snap[table] = {
                "columns": {
                    column["name"]: (
                        str(column["type"]),
                        bool(column["nullable"]),
                        None if column["default"] is None else str(column["default"]),
                    )
                    for column in inspector.get_columns(table)
                },
                "indexes": {
                    index["name"]: tuple(index["column_names"])
                    for index in inspector.get_indexes(table)
                },
                "fks": {
                    (tuple(fk["constrained_columns"]), fk["referred_table"])
                    for fk in inspector.get_foreign_keys(table)
                },
            }
        return snap

    async with adapter.engine.connect() as connection:
        return await connection.run_sync(reflect)


async def _alembic_version(adapter) -> str:
    async with adapter.engine.connect() as connection:
        return (
            await connection.execute(sa.text("SELECT version_num FROM alembic_version"))
        ).scalar()


async def _stamp(adapter, revision: str) -> None:
    """What ``stamp`` leaves behind — the bookkeeping row and nothing else."""
    async with adapter.engine.begin() as connection:
        await connection.execute(
            sa.text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await connection.execute(sa.text("DELETE FROM alembic_version"))
        await connection.execute(
            sa.text("INSERT INTO alembic_version (version_num) VALUES (:rev)"), {"rev": revision}
        )


async def _drop_columns(adapter, table: str, *columns: str, indexes: tuple = ()) -> None:
    """Damage a table the way a stale model surface would: batch mode, so
    SQLite drops columns that carry constraints too (an index on a dropped
    column has to go first — the rebuild would otherwise recreate it)."""

    def drop(sync_conn):
        with Operations.context(MigrationContext.configure(sync_conn)) as ops:
            with ops.batch_alter_table(table) as batch:
                for index in indexes:
                    batch.drop_index(index)
                for column in columns:
                    batch.drop_column(column)

    async with adapter.engine.begin() as connection:
        await connection.run_sync(drop)


async def _drop_table(adapter, table: str) -> None:
    async with adapter.engine.begin() as connection:
        await connection.execute(sa.text(f'DROP TABLE "{table}"'))


def _born():
    """A chain-born database at head, plus its snapshot."""
    adapter = _adapter()

    async def build():
        await adapter.create_database()
        return await _snapshot(adapter)

    return adapter, asyncio.run(build())


def test_chain_born_database_passes_through_as_a_no_op_also_twice_in_one_process():
    adapter, born = _born()
    assert asyncio.run(_alembic_version(adapter)) == _chain_head()
    assert ROUTER_ONLY_TABLES <= set(born)

    asyncio.run(_upgrade(adapter))
    assert asyncio.run(_snapshot(adapter)) == born

    second, second_born = _born()
    assert second_born == born


def test_partial_surface_legacy_database_converges_to_chain_born():
    """The pure-SDK 1.5.x shape: create_all of the frozen surface minus the
    router-only tables (what `import cognee` registered when such databases
    were born), stamped at the freeze point. The upgrade replays every
    post-freeze revision and then the reconcile — the path a real legacy
    database takes. Tables newer than the freeze must come from THEIR
    revisions, not from create_all: a model and its migration can disagree on
    what the name-level guard cannot see (a server default), and only the
    chain's version is the one every database actually has."""
    _register_all_models()
    from cognee.infrastructure.databases.relational import Base

    _, born = _born()
    legacy = _adapter()
    frozen = _frozen_table_names()

    async def build_legacy():
        partial = [
            t
            for t in Base.metadata.sorted_tables
            if t.name in frozen and t.name not in ROUTER_ONLY_TABLES
        ]
        async with legacy.engine.begin() as connection:
            await connection.run_sync(lambda c: Base.metadata.create_all(c, tables=partial))
        await _stamp(legacy, FREEZE_HEAD)
        before = await _snapshot(legacy)
        await _upgrade(legacy)
        return before, await _snapshot(legacy)

    before, after = asyncio.run(build_legacy())
    assert ROUTER_ONLY_TABLES.isdisjoint(before)
    assert ROUTER_ONLY_TABLES <= set(after)
    assert asyncio.run(_alembic_version(legacy)) == _chain_head()

    # The chain-only legacy table (no model) is the one thing a create_all-born
    # database legitimately lacks; the reconcile promises model truth only.
    assert set(born) - set(after) == {"notebooks"}
    for table in after:
        assert after[table]["columns"] == born[table]["columns"], table
        assert after[table]["indexes"] == born[table]["indexes"], table
        assert after[table]["fks"] == born[table]["fks"], table


def test_missing_columns_return_with_their_server_defaults_including_now():
    """``has_full_metrics`` (constant default) and ``created_at`` (``now()`` —
    SQLite's ADD COLUMN rejects it, so this exercises the batch rebuild)."""
    adapter, born = _born()

    async def damage_then_reconcile():
        await _drop_columns(adapter, "graph_metrics", "has_full_metrics", "created_at")
        await _stamp(adapter, PRE_RECONCILE_HEAD)
        damaged = await _snapshot(adapter)
        await _upgrade(adapter)
        healed = await _snapshot(adapter)
        async with adapter.engine.begin() as connection:
            await connection.execute(
                sa.text("INSERT INTO graph_metrics (id) VALUES (:id)"), {"id": uuid.uuid4().hex}
            )
            row = (
                await connection.execute(
                    sa.text("SELECT has_full_metrics, created_at FROM graph_metrics")
                )
            ).one()
        return damaged, healed, row

    damaged, healed, row = asyncio.run(damage_then_reconcile())
    assert "has_full_metrics" not in damaged["graph_metrics"]["columns"]
    assert "created_at" not in damaged["graph_metrics"]["columns"]
    assert healed["graph_metrics"]["columns"] == born["graph_metrics"]["columns"]
    assert row.has_full_metrics == 0 and row.created_at is not None


def test_missing_nullable_foreign_key_column_returns_with_its_constraint():
    adapter, born = _born()

    async def damage_then_reconcile():
        await _drop_columns(adapter, "users", "parent_user_id")
        await _stamp(adapter, PRE_RECONCILE_HEAD)
        damaged = await _snapshot(adapter)
        await _upgrade(adapter)
        return damaged, await _snapshot(adapter)

    damaged, healed = asyncio.run(damage_then_reconcile())
    assert (("parent_user_id",), "users") not in damaged["users"]["fks"]
    assert healed["users"]["columns"] == born["users"]["columns"]
    assert (("parent_user_id",), "users") in healed["users"]["fks"]


def test_not_null_column_missing_from_a_populated_table_aborts_before_any_change():
    """``permissions.name`` is NOT NULL with no server default; with a row in
    the table it cannot be added back without a backfill — the migration must
    refuse up front, and nothing else pending (a dropped table) may change."""
    from cognee.modules.migrations.startup import MigrationError

    adapter, _ = _born()

    async def populate_damage_reconcile():
        async with adapter.engine.begin() as connection:
            await connection.execute(
                sa.text("INSERT INTO permissions (id, name) VALUES (:id, 'read')"),
                {"id": uuid.uuid4().hex},
            )
        await _drop_columns(adapter, "permissions", "name", indexes=("ix_permissions_name",))
        await _drop_table(adapter, "sync_operations")
        await _stamp(adapter, PRE_RECONCILE_HEAD)
        try:
            await _upgrade(adapter)
        except MigrationError as error:
            # run_relational_migrations wraps; the reconcile's own message is the cause.
            return str(error.__cause__), await _snapshot(adapter), await _alembic_version(adapter)
        raise AssertionError("the reconcile must refuse a NOT NULL column it cannot backfill")

    message, after, version = asyncio.run(populate_damage_reconcile())
    assert "permissions.name" in message
    assert "sync_operations" not in after  # aborted BEFORE creating the missing table
    assert "name" not in after["permissions"]["columns"]
    assert version == PRE_RECONCILE_HEAD
