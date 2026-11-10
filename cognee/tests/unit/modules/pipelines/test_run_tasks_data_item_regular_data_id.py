"""Without incremental loading, an item's result entry still names its data items.

``add(incremental_loading=False)`` runs each input through the regular path,
which looks nothing up: the ids come from the ``Data`` rows ``ingest_data``
returns, or from the item itself when it already is a ``Data`` row (cognify).
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.data.models import Data
from cognee.modules.pipelines.operations import run_tasks_data_item as module


@pytest.mark.asyncio
async def test_an_added_input_reports_the_data_item_it_was_stored_as(monkeypatch):
    stored = _data_row()

    entry = await _final_entry(monkeypatch, data_item="some text", task_results=[[stored]])

    assert entry["data_id"] == stored.id
    assert "data_ids" not in entry


@pytest.mark.asyncio
async def test_an_input_stored_as_several_data_items_lists_them_all(monkeypatch):
    first, second = _data_row(), _data_row()

    entry = await _final_entry(
        monkeypatch, data_item="dlt source", task_results=[[first, second], [second]]
    )

    assert entry["data_id"] == first.id
    assert entry["data_ids"] == [first.id, second.id]


@pytest.mark.asyncio
async def test_a_data_row_reports_its_own_id_and_other_inputs_may_have_none(monkeypatch):
    item = _data_row()

    own = await _final_entry(monkeypatch, data_item=item, task_results=[[_data_row()]])
    none = await _final_entry(monkeypatch, data_item="text", task_results=[["a chunk"]])

    assert own["data_id"] == item.id
    assert "data_ids" not in own
    assert none["data_id"] is None


async def _final_entry(monkeypatch, data_item, task_results):
    async def run_tasks(**kwargs):
        for result in task_results:
            yield result

    monkeypatch.setattr(module, "run_tasks_with_telemetry", run_tasks)
    events = module.run_tasks_data_item_regular(
        data_item, SimpleNamespace(id=uuid4(), name="ds"), [], "pipeline", uuid4(), None, None
    )
    return [event async for event in events][-1]


def _data_row() -> Data:
    return Data(id=uuid4(), name="doc")
