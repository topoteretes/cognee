"""The ids of a dlt table's rows do not depend on how many rows the table has (SDK-1169).

``add()`` derives every row's ``data_id`` before ingestion. A table past SQLite's bind-parameter
limit used to fail there with "too many SQL variables" after dlt had already committed its
staging rows, and the derivation opened two database sessions per row. The ids themselves are
part of every ``data_id``, so batching must not change a single one.
"""

import importlib
import json
import math
from types import SimpleNamespace
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy import event, insert

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.methods.get_existing_data_ids import DATA_ID_BATCH_SIZE
from cognee.modules.data.models import Data
from cognee.tasks.ingestion.dlt_row_data import DltRowData
from cognee.tasks.ingestion.resolve_dlt_sources import (
    _build_source_manifest_item,
    _dlt_row_identifier,
    _stable_row_ids,
)

unique_id_module = importlib.import_module("cognee.modules.data.methods.get_unique_data_id")

USER = SimpleNamespace(id=UUID(int=1), tenant_id=UUID(int=2))
DATASET = UUID(int=3)
OTHER_DATASET = UUID(int=4)

# SQLite refuses a statement with more than 32,766 bind variables.
ROWS_ABOVE_THE_BIND_LIMIT = 40_000
# 10k Apollo contacts re-synced in 58 s, 42 s of it in per-row sessions.
RESYNC_ROWS = 10_000

# Derived with the one-session-per-row code this replaced: row "dlt:contacts:{i}:hash{i}" for
# USER in DATASET, then the same row without the dataset (the pre-scoping derivation).
ROW_0_ID = UUID("d4c00d29-2597-5877-901c-f1f25537f8df")
ROW_0_PRE_SCOPING_ID = UUID("c4562847-d685-5374-bb31-aaea40310868")
ROW_1_ID = UUID("e8388851-8635-5106-9741-7a8cc972a2bf")


def _row(index: int) -> DltRowData:
    return DltRowData(
        table_name="contacts",
        primary_key_columns=["id"],
        primary_key_value=str(index),
        row_data={"id": index},
        content_hash=f"hash{index}",
        schema_info=[],
        schema_hash="schema",
        foreign_keys=[],
        dlt_db_name="dlt_db",
        dataset_name="ds",
    )


async def _store(engine, ids, dataset_id):
    values = [
        {
            "id": data_id,
            "dataset_id": dataset_id,
            "name": "row",
            "content_hash": str(data_id),
            "raw_data_location": "file:///row.txt",
            "owner_id": USER.id,
            "pipeline_status": {},
            "token_count": -1,
        }
        for data_id in ids
    ]
    async with engine.get_async_session() as session:
        # executemany: a multi-row VALUES would hit the bind limit this file is about.
        await session.execute(insert(Data.__table__), values)
        await session.commit()


@pytest_asyncio.fixture
async def relational(monkeypatch, tmp_path):
    """A real SQLite relational store that counts the sessions and statements it serves."""
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path / 'relational.db'}")
    await engine.create_database()
    served = {"sessions": 0, "statements": 0}

    open_session = engine.get_async_session

    def counting_session():
        served["sessions"] += 1
        return open_session()

    engine.get_async_session = counting_session
    event.listen(
        engine.engine.sync_engine,
        "before_cursor_execute",
        lambda *_: served.__setitem__("statements", served["statements"] + 1),
    )
    monkeypatch.setattr(unique_id_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(
        "cognee.infrastructure.databases.relational.get_relational_engine", lambda: engine
    )
    return SimpleNamespace(engine=engine, served=served)


@pytest.mark.asyncio
async def test_a_table_above_the_bind_limit_gets_one_id_per_row(relational):
    # The last row repeats the first: dlt child tables can hold byte-identical rows, and
    # add() pairs ids with rows by position, so a repeat must keep its own id.
    rows = [_row(index) for index in range(ROWS_ABOVE_THE_BIND_LIMIT)] + [_row(0)]

    ids = await _stable_row_ids(rows, USER, DATASET)

    assert len(ids) == len(rows)
    assert len(set(ids)) == ROWS_ABOVE_THE_BIND_LIMIT
    assert ids[:2] == [ROW_0_ID, ROW_1_ID]
    assert ids[-1] == ROW_0_ID


@pytest.mark.asyncio
async def test_a_row_keeps_its_pre_scoping_id_only_in_its_own_dataset(relational):
    """Rows ingested before ids were dataset-namespaced are adopted in place, so an upgrade does
    not re-mint them. Row 0 has such a row in this dataset, row 1 only in another dataset."""
    row_0, row_1 = _row(0), _row(1)
    await _store(relational.engine, [ROW_0_PRE_SCOPING_ID], DATASET)
    row_1_pre_scoping_id = await unique_id_module.get_unique_data_id(
        _dlt_row_identifier(row_1), USER
    )
    await _store(relational.engine, [row_1_pre_scoping_id], OTHER_DATASET)

    ids = await _stable_row_ids([row_0, row_1], USER, DATASET)

    assert ids == [ROW_0_PRE_SCOPING_ID, ROW_1_ID]


@pytest.mark.asyncio
async def test_an_unchanged_resync_reads_the_database_per_batch_not_per_row(relational):
    rows = [_row(index) for index in range(RESYNC_ROWS)]
    first_ids = await _stable_row_ids(rows, USER, DATASET)
    await _store(relational.engine, first_ids, DATASET)
    relational.served.update(sessions=0, statements=0)

    ids = await _stable_row_ids(rows, USER, DATASET)

    assert ids == first_ids
    # Two derivations and the adoption probe, each one statement per batch of ids, all in the
    # one session: under NullPool every extra session is a fresh connection.
    batches = math.ceil(RESYNC_ROWS / DATA_ID_BATCH_SIZE)
    assert relational.served["statements"] <= 3 * batches
    assert relational.served["sessions"] == 1


@pytest.mark.asyncio
async def test_a_manifest_row_keeps_the_node_id_of_its_own_row(relational):
    """The manifest path derives its node ids in one batch too; each must stay on its row."""
    rows = [_row(index) for index in range(3)]

    manifest = await _build_source_manifest_item(rows, "contacts", "ds", USER)

    node_ids = {
        manifest_row["primary_key_value"]: manifest_row["node_id"]
        for manifest_row in json.loads(manifest.data)["rows"]
    }
    assert node_ids["0"] == str(ROW_0_PRE_SCOPING_ID)
    assert node_ids == {
        str(index): str(await unique_id_module.get_unique_data_id(_dlt_row_identifier(row), USER))
        for index, row in enumerate(rows)
    }
