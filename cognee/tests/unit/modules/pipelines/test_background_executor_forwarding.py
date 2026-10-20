"""The background executor keeps one subscriber informed across a chain of runs (SDK-397)."""

from types import SimpleNamespace

import pytest

from cognee.modules.pipelines.layers import pipeline_execution_mode
from cognee.modules.pipelines.layers.pipeline_execution_mode import (
    run_pipeline_as_background_process,
)


class _Info:
    def __init__(self, run_id, dataset_id="ds1"):
        self.pipeline_run_id = run_id
        self.dataset_id = dataset_id
        self.payload = None


@pytest.mark.asyncio
async def test_later_runs_of_a_chain_are_pushed_to_the_first_runs_queue(monkeypatch):
    pushed = []
    monkeypatch.setattr(
        pipeline_execution_mode, "push_to_queue", lambda run_id, info: pushed.append((run_id, info))
    )

    async def chain(**kwargs):
        yield _Info("caching-run")  # handed back as the started run
        yield _Info("caching-run")
        yield _Info("indexing-run")
        yield _Info("cognify-run")

    started = await run_pipeline_as_background_process(chain, datasets="ds1")
    assert started["ds1"].pipeline_run_id == "caching-run"

    task = next(iter(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS))
    await task

    by_queue = {}
    for run_id, info in pushed:
        by_queue.setdefault(run_id, []).append(info.pipeline_run_id)
    # The first run's own event is pushed once; every later run's event lands
    # on its own queue and on the first run's queue.
    assert by_queue["caching-run"] == ["caching-run", "indexing-run", "cognify-run"]
    assert by_queue["indexing-run"] == ["indexing-run"]
    assert by_queue["cognify-run"] == ["cognify-run"]


@pytest.mark.asyncio
async def test_a_single_run_is_pushed_once_per_event(monkeypatch):
    pushed = []
    monkeypatch.setattr(
        pipeline_execution_mode, "push_to_queue", lambda run_id, info: pushed.append(run_id)
    )

    async def single(**kwargs):
        yield _Info("only-run")
        yield _Info("only-run")
        yield _Info("only-run")

    await run_pipeline_as_background_process(single, datasets="ds1")
    task = next(iter(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS))
    await task

    assert pushed == ["only-run", "only-run"]
