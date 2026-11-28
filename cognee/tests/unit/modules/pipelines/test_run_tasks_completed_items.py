"""A run reads its items' state per batch, not per item (SDK-1715).

A dlt sync hands ``add()`` one pinned ``DataItem`` per staged row, changed or
not, and ``cognify()`` then runs over every ``Data`` row of the dataset. Each
item used to be looked up in its own session: on SQLite 31 s (add) and 13 s
(cognify) for 10,000 unchanged rows, 735 s and 80 s for 50,000. The add grew
with the square of the rows, since a pin's query read the whole dataset.
"""

import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import event, insert, select

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
)
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.ingestion.data_item import DataItem

run_tasks_module = importlib.import_module("cognee.modules.pipelines.operations.run_tasks")
item_module = importlib.import_module("cognee.modules.pipelines.operations.run_tasks_data_item")
telemetry_module = importlib.import_module(
    "cognee.modules.pipelines.operations.run_tasks_with_telemetry"
)

PIPELINE = "add_pipeline"
DATASET = UUID(int=1)
USER = SimpleNamespace(id=UUID(int=2), tenant_id=None)
COMPLETED = {PIPELINE: {str(DATASET): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}}


@asynccontextmanager
async def _no_database_context(*_args, **_kwargs):
    yield


@pytest_asyncio.fixture
async def relational(monkeypatch, tmp_path):
    """A real SQLite relational store that counts the sessions and statements it serves.

    The run's own bookkeeping (run log, telemetry, graph context) is stubbed, so
    the counts are the reads of the items' state.
    """
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path / 'relational.db'}")
    await engine.create_database()
    async with engine.get_async_session() as session:
        session.add(Dataset(id=DATASET, name="ds", owner_id=USER.id))
        await session.commit()

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
    for module in (run_tasks_module, item_module):
        monkeypatch.setattr(module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(run_tasks_module, "get_graph_engine", AsyncMock(return_value=object()))
    monkeypatch.setattr(
        run_tasks_module, "set_database_global_context_variables", _no_database_context
    )
    monkeypatch.setattr(
        run_tasks_module,
        "log_pipeline_run_start",
        AsyncMock(return_value=SimpleNamespace(pipeline_run_id=uuid4())),
    )
    for name in (
        "log_pipeline_run_complete",
        "log_pipeline_run_error",
        "log_pipeline_run_progress",
    ):
        monkeypatch.setattr(run_tasks_module, name, AsyncMock())
    for module in (run_tasks_module, telemetry_module):
        monkeypatch.setattr(module, "send_telemetry", lambda *_args, **_kwargs: None)
    return SimpleNamespace(engine=engine, served=served)


async def _store(engine, count, pipeline_status):
    ids = [uuid4() for _ in range(count)]
    values = [
        {
            "id": data_id,
            "dataset_id": DATASET,
            "name": f"row {index}",
            "label": f"label {index}",
            "external_metadata": {"_cognee": {"source_uri": f"https://example.test/{index}"}},
            "content_hash": str(data_id),
            "raw_data_location": "file:///row.txt",
            "owner_id": USER.id,
            "pipeline_status": pipeline_status,
            "token_count": -1,
        }
        for index, data_id in enumerate(ids)
    ]
    async with engine.get_async_session() as session:
        await session.execute(insert(Data.__table__), values)
        await session.commit()
    return ids


async def _rows(engine, ids):
    async with engine.get_async_session() as session:
        rows = (await session.scalars(select(Data).where(Data.id.in_(ids)))).all()
    by_id = {row.id: row for row in rows}
    return [by_id[data_id] for data_id in ids]


async def _run(relational, data, incremental_loading=True, data_cache=False):
    """Run ``data`` over DATASET: its item results, the reads it made, the items its task saw."""
    processed = []

    async def record(items):
        processed.extend(items)
        return items

    relational.served.update(sessions=0, statements=0)
    runs = [
        run_info
        async for run_info in run_tasks_module.run_tasks(
            [Task(record)],
            DATASET,
            data,
            USER,
            PIPELINE,
            incremental_loading=incremental_loading,
            data_cache=data_cache,
        )
    ]
    assert isinstance(runs[-1], PipelineRunCompleted), runs[-1]
    return runs[-1].data_ingestion_info, dict(relational.served), processed


def _already_completed(results, ids):
    """Each result is the skip entry of its stored row, as rollback and data_ids read it."""
    assert all(isinstance(result["run_info"], PipelineRunAlreadyCompleted) for result in results)
    assert [{key: result[key] for key in result if key != "run_info"} for result in results] == [
        {
            "data_id": data_id,
            "data_name": f"row {index}",
            "data_location": f"https://example.test/{index}",
            "data_label": f"label {index}",
        }
        for index, data_id in enumerate(ids)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flags",
    [{"incremental_loading": True}, {"incremental_loading": False, "data_cache": True}],
    ids=["incremental_loading", "data_cache"],
)
async def test_completed_pinned_items_are_read_per_batch_not_per_item(relational, flags):
    ids = await _store(relational.engine, 200, COMPLETED)
    items = [DataItem(data="unchanged", data_id=data_id) for data_id in ids]

    _, few, _ = await _run(relational, items[:20], **flags)
    results, many, processed = await _run(relational, items, **flags)

    assert many == few
    assert processed == []
    _already_completed(results, ids)


@pytest.mark.asyncio
async def test_completed_data_rows_are_not_read_again_per_item(relational):
    """cognify passes the rows run_pipeline_per_dataset read under the dataset lock."""
    ids = await _store(relational.engine, 200, COMPLETED)
    rows = await _rows(relational.engine, ids)

    _, few, _ = await _run(relational, rows[:20])
    results, many, processed = await _run(relational, rows)

    assert many == few
    assert processed == []
    _already_completed(results, ids)


@pytest.mark.asyncio
async def test_only_the_items_not_completed_are_processed_and_marked(relational):
    [done] = await _store(relational.engine, 1, COMPLETED)
    [pending] = await _store(relational.engine, 1, {})
    items = [DataItem(data="done", data_id=done), DataItem(data="pending", data_id=pending)]

    results, _, processed = await _run(relational, items)

    assert [item.data_id for item in processed] == [pending]
    assert [type(result["run_info"]) for result in results] == [
        PipelineRunAlreadyCompleted,
        PipelineRunCompleted,
    ]
    [row] = await _rows(relational.engine, [pending])
    assert row.pipeline_status == COMPLETED
