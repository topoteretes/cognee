"""A pinned legacy data id is reported as the data item it was forked into.

``ingest_data`` stores a ``DataItem`` with a pinned ``data_id`` under
``resolve_data_id(dataset_id, pin)``, which maps a legacy id to the canonical
data item. The incremental pipeline must resolve the pin the same way, or it
looks up, marks and reports an id that has no data item behind it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.run_tasks_data_item as item_module
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.tasks.ingestion.data_item import DataItem


@pytest.mark.asyncio
async def test_a_pinned_legacy_id_is_reported_as_its_canonical_data_item(monkeypatch):
    legacy, canonical = uuid4(), uuid4()
    dataset = SimpleNamespace(id=uuid4(), name="ds")
    # The canonical data item was already processed, so the item is skipped.
    stored = SimpleNamespace(
        id=canonical,
        pipeline_status={"p": {str(dataset.id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}},
        name="doc",
        label=None,
        external_metadata={},
    )
    engine = _Engine(query_results=[[stored]])
    monkeypatch.setattr(item_module, "get_relational_engine", lambda: engine)

    outputs = [
        out
        async for out in item_module.run_tasks_data_item_incremental(
            data_item=DataItem(data="text", data_id=legacy),
            dataset=dataset,
            tasks=[],
            pipeline_name="p",
            pipeline_id="p",
            pipeline_run_id=uuid4(),
            ctx=SimpleNamespace(extras={}),
            user=SimpleNamespace(id=uuid4()),
        )
    ]

    assert outputs[-1]["data_id"] == canonical
    # Resolving the pin shares the item's one pre-check session.
    assert engine.sessions_opened == 1


@pytest.mark.asyncio
async def test_the_exact_id_wins_over_a_data_item_forked_from_it():
    pin = uuid4()
    forked, exact = SimpleNamespace(id=uuid4()), SimpleNamespace(id=pin)
    session = _Session(query_results=[[forked, exact]])

    assert await item_module._find_pinned_data_item(session, uuid4(), pin) is exact


@pytest.mark.asyncio
async def test_an_id_unknown_in_the_dataset_is_looked_up_alone():
    pin = uuid4()
    elsewhere = SimpleNamespace(id=pin)
    session = _Session(query_results=[[], [elsewhere]])

    assert await item_module._find_pinned_data_item(session, uuid4(), pin) is elsewhere


@pytest.mark.asyncio
async def test_failed_item_can_be_retried_and_only_success_is_skipped(monkeypatch):
    """Provider recovery must not require deleting an input or forcing all items."""
    dataset = SimpleNamespace(id=uuid4(), name="ds")
    stored = item_module.Data(id=uuid4(), name="doc", pipeline_status={})
    session = _Session([[stored], [stored], [stored], [stored]])
    session.merge = AsyncMock()
    session.commit = AsyncMock()
    engine = SimpleNamespace(get_async_session=lambda: session)
    monkeypatch.setattr(item_module, "get_relational_engine", lambda: engine)
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "false")
    attempts = 0

    async def run(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError()
        yield "processed"

    monkeypatch.setattr(item_module, "run_tasks_with_telemetry", run)

    async def invoke():
        return [
            out
            async for out in item_module.run_tasks_data_item_incremental(
                data_item=stored,
                dataset=dataset,
                tasks=[],
                pipeline_name="p",
                pipeline_id="p",
                pipeline_run_id=uuid4(),
                ctx=SimpleNamespace(extras={}),
                user=SimpleNamespace(id=uuid4()),
            )
        ]

    failed = await invoke()
    assert isinstance(failed[-1]["error"], TimeoutError)
    assert stored.pipeline_status == {}
    session.commit.assert_not_awaited()
    await invoke()
    assert (
        stored.pipeline_status["p"][str(dataset.id)]
        == DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
    )
    skipped = await invoke()
    assert isinstance(skipped[-1]["run_info"], item_module.PipelineRunAlreadyCompleted)
    assert attempts == 2
    session.commit.assert_awaited_once()


class _Engine:
    """A relational engine whose queries return ``query_results`` in order."""

    def __init__(self, query_results):
        self.query_results = list(query_results)
        self.sessions_opened = 0

    def get_async_session(self):
        self.sessions_opened += 1
        return _Session(self.query_results)


class _Session:
    def __init__(self, query_results):
        self.query_results = query_results

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def execute(self, statement):
        return _Result(self.query_results.pop(0))


class _Result:
    """One query's rows, read the ways the code under test reads them."""

    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return SimpleNamespace(all=lambda: self.rows)

    def scalar_one_or_none(self):
        return self.rows[0] if self.rows else None
