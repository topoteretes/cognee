"""The regular (non-incremental) item path reports the row an item was stored under.

``add(..., incremental_loading=False, data_cache=False)`` is the explicit
re-ingest path. Its per-item result used to carry no ``data_id`` at all, so
``PipelineRunInfo.data_ids`` came back empty for every row the call stored.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.data.models import Data
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunYield
from cognee.modules.pipelines.operations import run_tasks_data_item as module
from cognee.tasks.ingestion.data_item import DataItem


def _stream_of(*results):
    async def run_tasks_with_telemetry(**_kwargs):
        for result in results:
            yield result

    return run_tasks_with_telemetry


async def _final_entry(data_item, monkeypatch, *results):
    monkeypatch.setattr(module, "run_tasks_with_telemetry", _stream_of(*results))
    events = module.run_tasks_data_item_regular(
        data_item=data_item,
        dataset=SimpleNamespace(id=uuid4(), name="ds"),
        tasks=[],
        pipeline_id="add_pipeline",
        pipeline_run_id=str(uuid4()),
        ctx=None,
        user=object(),
    )
    collected = [event async for event in events]
    assert all(isinstance(event, PipelineRunYield) for event in collected[:-1])
    entry = collected[-1]
    assert isinstance(entry["run_info"], PipelineRunCompleted)
    return entry


@pytest.mark.asyncio
async def test_regular_path_reports_the_row_the_last_task_returned(monkeypatch):
    """add's ingest_data returns the Data rows it stored for the item."""
    row = Data(id=uuid4())

    entry = await _final_entry("some text", monkeypatch, [row])

    assert entry["data_id"] == row.id
    assert "data_ids" not in entry


@pytest.mark.asyncio
async def test_regular_path_keeps_the_id_a_data_row_already_has(monkeypatch):
    row = Data(id=uuid4())

    entry = await _final_entry(row, monkeypatch, ["whatever the tasks produced"])

    assert entry["data_id"] == row.id


@pytest.mark.asyncio
async def test_regular_path_keeps_the_id_a_data_item_was_pinned_to(monkeypatch):
    pinned = uuid4()
    item = DataItem(data="text", data_id=pinned)

    entry = await _final_entry(item, monkeypatch, [Data(id=uuid4())])

    assert entry["data_id"] == pinned


@pytest.mark.asyncio
async def test_regular_path_reports_no_id_when_no_row_surfaced(monkeypatch):
    """Pipelines whose items are not Data rows (custom tasks over plain
    values) still get their run_info; there is no row to name."""
    entry = await _final_entry("text", monkeypatch, ["not a row"], 42)

    assert entry["data_id"] is None
    assert "data_ids" not in entry


@pytest.mark.asyncio
async def test_regular_path_lists_every_row_when_an_item_spans_several(monkeypatch):
    first, second = Data(id=uuid4()), Data(id=uuid4())

    entry = await _final_entry("text", monkeypatch, [first, second], [first])

    assert entry["data_id"] == first.id
    assert entry["data_ids"] == [first.id, second.id]
