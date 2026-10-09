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

import cognee.modules.data.methods as data_methods
import cognee.modules.pipelines.operations.run_tasks_data_item as item_module
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.tasks.ingestion.data_item import DataItem


@pytest.mark.asyncio
async def test_a_pinned_legacy_id_is_reported_as_its_canonical_data_item(monkeypatch):
    legacy, canonical = uuid4(), uuid4()
    dataset = SimpleNamespace(id=uuid4(), name="ds")
    resolve = AsyncMock(return_value=canonical)
    monkeypatch.setattr(data_methods, "resolve_data_id", resolve)
    # The canonical data item was already processed, so the item is skipped.
    stored = SimpleNamespace(
        id=canonical,
        pipeline_status={"p": {str(dataset.id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}},
        name="doc",
        label=None,
        external_metadata={},
    )
    monkeypatch.setattr(item_module, "get_relational_engine", lambda: _engine_returning(stored))

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

    resolve.assert_awaited_once_with(dataset.id, legacy)
    assert outputs[-1]["data_id"] == canonical


def _engine_returning(row):
    """A relational engine whose every query finds ``row``."""

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def execute(self, statement):
            return SimpleNamespace(scalar_one_or_none=lambda: row)

    return SimpleNamespace(get_async_session=_Session)
