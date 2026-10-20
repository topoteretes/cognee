"""The background executor keeps one subscriber informed across a chain of runs (SDK-397)."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.pipelines.layers import pipeline_execution_mode
from cognee.modules.pipelines.layers.pipeline_execution_mode import (
    run_pipeline_as_background_process,
)
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunYield
from cognee.modules.pipelines.operations import run_tasks_data_item


class _Info:
    def __init__(self, run_id, chain_run_id=None, dataset_id="ds1"):
        self.pipeline_run_id = run_id
        self.chain_run_id = chain_run_id
        self.dataset_id = dataset_id
        self.payload = None


@pytest.mark.asyncio
async def test_events_marked_with_the_chain_run_go_to_its_queue_only(monkeypatch):
    pushed = []
    monkeypatch.setattr(
        pipeline_execution_mode, "push_to_queue", lambda run_id, info: pushed.append((run_id, info))
    )

    async def chain(**kwargs):
        yield _Info("cognify-run")  # handed back as the started run
        yield _Info("caching-run", chain_run_id="cognify-run")
        yield _Info("indexing-run", chain_run_id="cognify-run")
        yield _Info("cognify-run")

    started = await run_pipeline_as_background_process(chain, datasets="ds1")
    assert started["ds1"].pipeline_run_id == "cognify-run"

    task = next(iter(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS))
    await task

    # Every event after the handed-back one lands on the handed-back run's queue;
    # no queue is opened under the other runs' ids, which no caller holds.
    assert [(run_id, info.pipeline_run_id) for run_id, info in pushed] == [
        ("cognify-run", "caching-run"),
        ("cognify-run", "indexing-run"),
        ("cognify-run", "cognify-run"),
    ]


@pytest.mark.asyncio
async def test_unmarked_events_are_pushed_under_their_own_run(monkeypatch):
    pushed = []
    monkeypatch.setattr(
        pipeline_execution_mode, "push_to_queue", lambda run_id, info: pushed.append(run_id)
    )

    async def single(**kwargs):
        yield _Info("started-run")
        yield _Info("started-run")
        yield _Info("another-run")  # no chain marker: its own queue

    await run_pipeline_as_background_process(single, datasets="ds1")
    task = next(iter(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS))
    await task

    assert pushed == ["started-run", "another-run"]


COGNIFY_RUN = uuid4()
INDEXING_RUN = uuid4()


def _yielded():
    return PipelineRunYield(
        pipeline_run_id=INDEXING_RUN, dataset_id=uuid4(), dataset_name="ds", payload=None
    )


@pytest.mark.parametrize(
    ("extras", "queue", "chain_run_id"),
    [
        ({"chain_run_id": COGNIFY_RUN}, COGNIFY_RUN, COGNIFY_RUN),
        ({}, INDEXING_RUN, None),
    ],
)
def test_per_task_progress_follows_the_chain(monkeypatch, extras, queue, chain_run_id):
    """A preparation run's progress ticks reach the subscriber of the cognify run."""
    pushed = []
    monkeypatch.setattr(
        run_tasks_data_item, "push_to_queue", lambda run_id, info: pushed.append((run_id, info))
    )
    ctx = SimpleNamespace(task_sequence=["extract_chunks_from_documents"], extras=extras)

    run_tasks_data_item._push_stage_progress(_yielded(), ctx, [object(), object()], INDEXING_RUN)

    ((run_id, progress),) = pushed
    assert run_id == queue
    assert progress.pipeline_run_id == INDEXING_RUN
    assert progress.chain_run_id == chain_run_id
    assert (progress.current_stage, progress.stage_index, progress.stage_total) == (
        "extract_chunks_from_documents",
        1,
        2,
    )
