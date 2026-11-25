"""An empty data list runs a pipeline over no data; only None means the whole dataset.

An add whose document sources had nothing new resolves no items (SDK-1715). When an empty
list meant "every row of the dataset", that add ran its pipeline over all of them.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.locks import held_datasets

pipeline = importlib.import_module("cognee.modules.pipelines.operations.pipeline")


async def _run_body_data(monkeypatch, data):
    """The data ``run_pipeline_per_dataset`` hands ``run_tasks`` for this ``data``."""
    whole_dataset = [SimpleNamespace(id=uuid4())]
    monkeypatch.setattr(pipeline, "get_dataset_data", AsyncMock(return_value=whole_dataset))
    monkeypatch.setattr(
        pipeline, "get_selected_dataset_data", AsyncMock(side_effect=lambda _id, data: data)
    )
    seen = []

    async def run_tasks(_tasks, _dataset_id, body_data, *_args, **_kwargs):
        seen.append(body_data)
        yield "done"

    monkeypatch.setattr(pipeline, "run_tasks", run_tasks)
    dataset = SimpleNamespace(id=uuid4())
    # Held already, as in a nested run, so the test takes no dataset lock.
    token = held_datasets.set(frozenset({dataset.id}))
    try:
        async for _ in pipeline.run_pipeline_per_dataset(dataset, SimpleNamespace(), [], data):
            pass
    finally:
        held_datasets.reset(token)
    return seen[0], whole_dataset


@pytest.mark.asyncio
async def test_an_empty_list_runs_over_no_data(monkeypatch):
    body_data, _ = await _run_body_data(monkeypatch, [])

    assert body_data == []
    pipeline.get_dataset_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_data_runs_over_the_whole_dataset(monkeypatch):
    body_data, whole_dataset = await _run_body_data(monkeypatch, None)

    assert body_data == whole_dataset
