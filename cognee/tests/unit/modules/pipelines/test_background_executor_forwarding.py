"""The background executor keeps one subscriber informed across a chain of runs (SDK-397)."""

import pytest

from cognee.modules.pipelines.layers import pipeline_execution_mode
from cognee.modules.pipelines.layers.pipeline_execution_mode import (
    run_pipeline_as_background_process,
)


class _Info:
    def __init__(self, run_id, chain_run_id=None, dataset_id="ds1"):
        self.pipeline_run_id = run_id
        self.chain_run_id = chain_run_id
        self.dataset_id = dataset_id
        self.payload = None


@pytest.mark.asyncio
async def test_events_marked_with_the_chain_run_are_pushed_to_its_queue_too(monkeypatch):
    pushed = []
    monkeypatch.setattr(
        pipeline_execution_mode, "push_to_queue", lambda run_id, info: pushed.append((run_id, info))
    )

    async def chain(**kwargs):
        yield _Info("caching-run")  # handed back as the started run
        yield _Info("caching-run")
        yield _Info("indexing-run", chain_run_id="caching-run")
        yield _Info("cognify-run", chain_run_id="caching-run")

    started = await run_pipeline_as_background_process(chain, datasets="ds1")
    assert started["ds1"].pipeline_run_id == "caching-run"

    task = next(iter(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS))
    await task

    by_queue = {}
    for run_id, info in pushed:
        by_queue.setdefault(run_id, []).append(info.pipeline_run_id)
    # The first run's own event is pushed once; every marked event lands on its
    # own queue and on the first run's queue.
    assert by_queue["caching-run"] == ["caching-run", "indexing-run", "cognify-run"]
    assert by_queue["indexing-run"] == ["indexing-run"]
    assert by_queue["cognify-run"] == ["cognify-run"]


@pytest.mark.asyncio
async def test_unmarked_events_are_pushed_once_whatever_their_run_id(monkeypatch):
    pushed = []
    monkeypatch.setattr(
        pipeline_execution_mode, "push_to_queue", lambda run_id, info: pushed.append(run_id)
    )

    async def single(**kwargs):
        yield _Info("started-run")
        yield _Info("started-run")
        yield _Info("another-run")  # no chain marker: not forwarded

    await run_pipeline_as_background_process(single, datasets="ds1")
    task = next(iter(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS))
    await task

    assert pushed == ["started-run", "another-run"]
