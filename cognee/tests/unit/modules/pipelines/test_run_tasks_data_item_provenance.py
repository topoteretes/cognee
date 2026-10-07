import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

module = importlib.import_module("cognee.modules.pipelines.operations.run_tasks_data_item")


async def _completed_regular(**_kwargs):
    yield {"run_info": object()}


@pytest.mark.asyncio
async def test_provenance_flushes_once_at_data_item_completion(monkeypatch):
    flush = AsyncMock()
    monkeypatch.setattr(module, "run_tasks_data_item_regular", _completed_regular)
    monkeypatch.setattr(module, "flush_context_provenance", flush)
    ctx = SimpleNamespace()

    result = await module.run_tasks_data_item(
        data_item=object(),
        dataset=object(),
        tasks=[],
        pipeline_name="cognify_pipeline",
        pipeline_id="pipeline",
        pipeline_run_id="run",
        ctx=ctx,
        user=object(),
        incremental_loading=False,
        data_cache=False,
    )

    assert "run_info" in result
    flush.assert_awaited_once_with(ctx)
