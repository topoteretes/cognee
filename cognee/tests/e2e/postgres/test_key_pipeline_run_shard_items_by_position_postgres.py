"""Revision 9b4d7e2a6c1f on PostgreSQL: the item table is changed in place.

Rows written under the previous revision keep their place: a row's position is
its rank in its shard by ``data_id``, the order the shard's worker read its
items in. Runs in a schema of its own inside one transaction that is rolled
back, so nothing is left behind.

Skipped unless COGNEE_PG_TEST_URL names a PostgreSQL server, e.g.:

    COGNEE_PG_TEST_URL=postgres://cognee:cognee@localhost:5432/cognee_db
"""

import importlib.util
import os
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

PG_URL = os.getenv("COGNEE_PG_TEST_URL")

pytestmark = [
    pytest.mark.skipif(not PG_URL, reason="COGNEE_PG_TEST_URL not set"),
    pytest.mark.asyncio,
]

_VERSIONS = Path(__file__).resolve().parents[3] / "alembic" / "versions"


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
ITEMS, SHARDS = migration.ITEMS, migration.SHARDS


def _run(conn: sa.Connection, fn) -> None:
    with Operations.context(MigrationContext.configure(conn)):
        fn()


def _key(conn: sa.Connection) -> list[str]:
    return sa.inspect(conn).get_pk_constraint(ITEMS)["constrained_columns"]


def _nullable(conn: sa.Connection) -> dict[str, bool]:
    return {col["name"]: col["nullable"] for col in sa.inspect(conn).get_columns(ITEMS)}


def _rows(conn: sa.Connection, columns: str) -> list[tuple]:
    return [tuple(row) for row in conn.execute(sa.text(f"SELECT {columns} FROM {ITEMS}"))]


def _check(conn: sa.Connection) -> None:
    dataset, shards = uuid4(), [uuid4(), uuid4()]
    data = sorted(uuid4() for _ in range(5))
    conn.execute(sa.text("CREATE TABLE datasets (id uuid PRIMARY KEY)"))
    conn.execute(sa.text("CREATE TABLE data (id uuid PRIMARY KEY)"))
    _run(conn, previous.upgrade)
    conn.execute(sa.text("INSERT INTO datasets VALUES (:id)"), {"id": dataset})
    conn.execute(sa.text("INSERT INTO data VALUES (:id)"), [{"id": row} for row in data])
    conn.execute(
        sa.text(
            f"INSERT INTO {SHARDS} (id, pipeline_run_id, dataset_id, shard_index, status, spec, "
            "created_at) VALUES (:id, :run, :dataset, :index, 'finished', '{}', now())"
        ),
        [
            {"id": shard, "run": uuid4(), "dataset": dataset, "index": i}
            for i, shard in enumerate(shards)
        ],
    )
    # Three items in the first shard and two in the second, written out of order.
    held = {shards[0]: [data[3], data[0], data[4]], shards[1]: [data[2], data[1]]}
    conn.execute(
        sa.text(
            f"INSERT INTO {ITEMS} (shard_id, data_id, status) VALUES (:shard, :data, 'completed')"
        ),
        [{"shard": shard, "data": row} for shard, rows in held.items() for row in rows],
    )

    _run(conn, migration.upgrade)
    _run(conn, migration.upgrade)  # and again: nothing changes

    assert _key(conn) == ["shard_id", "item_index"]
    assert _nullable(conn)["data_id"] is True and _nullable(conn)["item_index"] is False
    # Every row is kept, at its rank in its shard by data_id.
    assert sorted(_rows(conn, "shard_id, item_index, data_id, status")) == sorted(
        (shard, position, row, "completed")
        for shard, rows in held.items()
        for position, row in enumerate(sorted(rows))
    )
    # An item that is a value has no data row.
    conn.execute(
        sa.text(
            f"INSERT INTO {ITEMS} (shard_id, item_index, value, status) "
            "VALUES (:shard, 3, '{\"note\": 1}', 'completed')"
        ),
        {"shard": shards[0]},
    )

    _run(conn, migration.downgrade)
    _run(conn, migration.downgrade)

    # The earlier shape again: its data rows are kept, the value's row is not.
    assert _key(conn) == ["shard_id", "data_id"]
    assert set(_nullable(conn)) == {"shard_id", "data_id", "status", "error_class", "error_message"}
    assert sorted(_rows(conn, "shard_id, data_id")) == sorted(
        (shard, row) for shard, rows in held.items() for row in rows
    )


async def test_rows_written_under_the_previous_revision_keep_their_place():
    engine = create_async_engine(make_url(PG_URL).set(drivername="postgresql+asyncpg"))
    schema = f"shard_items_{uuid4().hex[:10]}"
    try:
        async with engine.connect() as conn:
            transaction = await conn.begin()
            try:
                await conn.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
                await conn.execute(sa.text(f'SET LOCAL search_path TO "{schema}"'))
                await conn.run_sync(_check)
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()
