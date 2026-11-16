"""Revision 9b4d7e2a6c1f keys pipeline_run_shard_items by an item's position.

On a database that is not PostgreSQL the table is empty (no run hands shards
out there) and is made again in its new shape. These tests run that path on
real in-memory SQLite, from the states an interruption can leave behind; the
PostgreSQL path, which changes the table in place and keeps its rows, is in
``tests/e2e/postgres/test_key_pipeline_run_shard_items_by_position_postgres.py``.
"""

import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

from cognee.modules.pipelines.models import PipelineRunShardItem

_VERSIONS = Path(__file__).resolve().parents[2] / "alembic" / "versions"


def _load(file_name: str):
    spec = importlib.util.spec_from_file_location(
        file_name.removesuffix(".py"), _VERSIONS / file_name
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = _load("3cb1c5e71f26_add_pipeline_run_shards.py")
migration = _load("9b4d7e2a6c1f_key_pipeline_run_shard_items_by_position.py")
ITEMS = migration.ITEMS


def _run(engine: sa.Engine, fn) -> None:
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            fn()


def _engine_at_the_previous_revision() -> sa.Engine:
    engine = sa.create_engine("sqlite://")
    meta = sa.MetaData()
    sa.Table("datasets", meta, sa.Column("id", sa.Uuid, primary_key=True))
    sa.Table("data", meta, sa.Column("id", sa.Uuid, primary_key=True))
    meta.create_all(engine)
    _run(engine, previous.upgrade)
    return engine


def _shape(engine: sa.Engine) -> dict[str, tuple[str, bool, int]]:
    """Each column's type, whether it may be NULL, and its place in the primary key."""
    return {
        col["name"]: (str(col["type"]), col["nullable"], col["primary_key"])
        for col in sa.inspect(engine).get_columns(ITEMS)
    }


def _indexes(engine: sa.Engine) -> set[str]:
    return {idx["name"] for idx in sa.inspect(engine).get_indexes(ITEMS) if idx["name"]}


def _references(engine: sa.Engine) -> set[tuple[str, str, str]]:
    return {
        (key["constrained_columns"][0], key["referred_table"], key["options"].get("ondelete"))
        for key in sa.inspect(engine).get_foreign_keys(ITEMS)
    }


def test_upgrade_gives_the_item_table_the_shape_its_model_declares():
    engine = _engine_at_the_previous_revision()
    previous_shape = _shape(engine)

    _run(engine, migration.upgrade)

    from_model = sa.create_engine("sqlite://")
    PipelineRunShardItem.__table__.create(from_model)
    assert _shape(engine) == _shape(from_model) != previous_shape
    assert _indexes(engine) == {index.name for index in PipelineRunShardItem.__table__.indexes}
    # An item row still goes with its shard, and with its data row when it is one.
    assert _references(engine) == {
        ("shard_id", migration.SHARDS, "CASCADE"),
        ("data_id", "data", "CASCADE"),
    }


def test_upgrade_is_idempotent():
    engine = _engine_at_the_previous_revision()

    _run(engine, migration.upgrade)
    shape = _shape(engine)
    _run(engine, migration.upgrade)

    assert _shape(engine) == shape
    assert _indexes(engine) == {migration.DATA_ID_INDEX}


def test_upgrade_finishes_after_an_interruption_between_dropping_and_creating():
    engine = _engine_at_the_previous_revision()
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP TABLE {ITEMS}"))

    _run(engine, migration.upgrade)

    assert "item_index" in _shape(engine)
    assert _indexes(engine) == {migration.DATA_ID_INDEX}


def test_a_table_with_rows_is_refused_and_keeps_them():
    """Nothing writes item rows on a database that is not PostgreSQL, so there is
    no rule for keeping them: the revision stops instead of emptying the table."""
    engine = _engine_at_the_previous_revision()
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                f"INSERT INTO {ITEMS} (shard_id, data_id, status) VALUES (:shard, :data, 'pending')"
            ),
            {"shard": uuid4().hex, "data": uuid4().hex},
        )

    with pytest.raises(RuntimeError, match="holds 1 rows"):
        _run(engine, migration.upgrade)

    with engine.connect() as conn:
        assert conn.execute(sa.text(f"SELECT count(*) FROM {ITEMS}")).scalar() == 1
    assert "item_index" not in _shape(engine)


def test_downgrade_restores_the_previous_shape_and_can_be_repeated():
    engine = _engine_at_the_previous_revision()
    previous_shape = _shape(engine)
    _run(engine, migration.upgrade)

    _run(engine, migration.downgrade)
    _run(engine, migration.downgrade)

    assert _shape(engine) == previous_shape
    assert _indexes(engine) == {migration.DATA_ID_INDEX}
