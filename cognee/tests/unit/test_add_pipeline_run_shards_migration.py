"""Revision 3cb1c5e71f26 adds pipeline_run_shards and pipeline_run_shard_items.

SQLite DDL is not transactional, so a process that dies between two statements
leaves the earlier ones in place. Every table and index of the revision is
guarded on its own, and these tests rerun it from the states an interruption
can leave behind. Real in-memory SQLite, no cognee engine.
"""

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

from cognee.modules.pipelines.models import PipelineRunShard, PipelineRunShardItem

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "3cb1c5e71f26_add_pipeline_run_shards.py"
)
_SPEC = importlib.util.spec_from_file_location("pipeline_run_shards_migration", _MIGRATION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)

SHARDS, ITEMS = migration.SHARDS, migration.ITEMS


def _engine_before_upgrade() -> sa.Engine:
    """The two tables the new ones reference, as the previous head has them."""
    engine = sa.create_engine("sqlite://")
    meta = sa.MetaData()
    sa.Table("datasets", meta, sa.Column("id", sa.Uuid, primary_key=True))
    sa.Table("data", meta, sa.Column("id", sa.Uuid, primary_key=True))
    meta.create_all(engine)
    return engine


def _run(engine: sa.Engine, fn) -> None:
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            fn()


def _columns(engine: sa.Engine, table: str) -> set[str]:
    return {col["name"] for col in sa.inspect(engine).get_columns(table)}


def _indexes(engine: sa.Engine, table: str) -> set[str]:
    return {idx["name"] for idx in sa.inspect(engine).get_indexes(table) if idx["name"]}


def _shape(engine: sa.Engine, table: str) -> dict[str, tuple[str, bool, int]]:
    """Each column's type, whether it may be NULL, and its place in the primary key."""
    return {
        col["name"]: (str(col["type"]), col["nullable"], col["primary_key"])
        for col in sa.inspect(engine).get_columns(table)
    }


def _references(engine: sa.Engine, table: str) -> set[tuple[str, str, str]]:
    """(column, referenced table, what a delete of the referenced row does)."""
    return {
        (key["constrained_columns"][0], key["referred_table"], key["options"].get("ondelete"))
        for key in sa.inspect(engine).get_foreign_keys(table)
    }


def test_upgrade_creates_both_tables_as_the_models_declare_them():
    engine = _engine_before_upgrade()

    _run(engine, migration.upgrade)

    # The same columns as the models: a column on one side only is the gap the
    # lockstep guard exists to catch.
    assert _columns(engine, SHARDS) == set(PipelineRunShard.__table__.columns.keys())
    assert _columns(engine, ITEMS) == set(PipelineRunShardItem.__table__.columns.keys())
    assert _indexes(engine, SHARDS) == set(migration.SHARD_INDEXES)
    assert _indexes(engine, ITEMS) == set(migration.ITEM_INDEXES)
    assert {index.name for index in PipelineRunShard.__table__.indexes} == set(
        migration.SHARD_INDEXES
    )
    assert {index.name for index in PipelineRunShardItem.__table__.indexes} == set(
        migration.ITEM_INDEXES
    )
    # And the same types, NULL rules and primary keys as the tables the models
    # create by themselves.
    from_models = sa.create_engine("sqlite://")
    for table in (PipelineRunShard.__table__, PipelineRunShardItem.__table__):
        table.create(from_models)
        assert _shape(engine, table.name) == _shape(from_models, table.name)
    # A shard goes with its dataset, an item row with its shard and its data item.
    assert _references(engine, SHARDS) == {("dataset_id", "datasets", "CASCADE")}
    assert _references(engine, ITEMS) == {
        ("shard_id", SHARDS, "CASCADE"),
        ("data_id", "data", "CASCADE"),
    }


def test_upgrade_is_idempotent():
    engine = _engine_before_upgrade()

    _run(engine, migration.upgrade)
    _run(engine, migration.upgrade)

    assert _indexes(engine, SHARDS) == set(migration.SHARD_INDEXES)
    assert sa.inspect(engine).has_table(ITEMS)


def test_upgrade_finishes_after_an_interruption_before_the_indexes():
    """The first table exists without its indexes, the second not at all."""
    engine = _engine_before_upgrade()
    _run(engine, migration.upgrade)
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP TABLE {ITEMS}"))
        for name in migration.SHARD_INDEXES:
            conn.execute(sa.text(f"DROP INDEX {name}"))

    _run(engine, migration.upgrade)

    assert _indexes(engine, SHARDS) == set(migration.SHARD_INDEXES)
    assert _indexes(engine, ITEMS) == set(migration.ITEM_INDEXES)


def test_downgrade_removes_both_tables_and_can_be_repeated():
    engine = _engine_before_upgrade()
    _run(engine, migration.upgrade)

    _run(engine, migration.downgrade)
    _run(engine, migration.downgrade)

    inspector = sa.inspect(engine)
    assert not inspector.has_table(SHARDS) and not inspector.has_table(ITEMS)
    assert inspector.has_table("datasets") and inspector.has_table("data")


def test_downgrade_finishes_after_an_interruption_between_the_tables():
    engine = _engine_before_upgrade()
    _run(engine, migration.upgrade)
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP TABLE {ITEMS}"))

    _run(engine, migration.downgrade)

    assert not sa.inspect(engine).has_table(SHARDS)
