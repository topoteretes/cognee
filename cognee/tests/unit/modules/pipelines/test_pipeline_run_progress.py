"""In-flight progress signal for add/cognify pipelines (CLO-557).

Covered here:
- run_tasks.py calls log_pipeline_run_progress for completed items, with a
  monotonically increasing completed_items count out of the correct total —
  regardless of gather() interleaving order.
- An item that hard-raises is still counted (finally-based, not a plain
  post-return increment).
- Large batches throttle the DB write itself, not just the in-memory count.
- log_pipeline_run_progress updates the run's one STARTED row in place
  instead of inserting a new pipeline_runs row per tick, so a long batch
  doesn't grow the table unboundedly.
- A tick arriving after the run already reached a terminal status is
  dropped, not inserted as a spurious new STARTED row (future-proofing —
  today's only caller, run_tasks.py, can't actually trigger this).
- pipeline_run_info_queues correctly round-trips a PipelineRunProgress event,
  the same way it already does for the other PipelineRunInfo subclasses.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import select

import cognee.modules.pipelines.operations.run_tasks as run_tasks_module
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunProgress
from cognee.modules.pipelines.operations.log_pipeline_run_progress import (
    log_pipeline_run_progress,
)
from cognee.modules.pipelines.operations.log_pipeline_run_complete import (
    log_pipeline_run_complete,
)
from cognee.modules.pipelines.operations.log_pipeline_run_start import log_pipeline_run_start
from cognee.modules.pipelines.queues import pipeline_run_info_queues as queues_module
from cognee.infrastructure.databases.relational import (
    create_db_and_tables,
    get_relational_engine,
)


@pytest.mark.asyncio
async def test_run_tasks_logs_progress_once_per_item(monkeypatch, runner_plumbing):
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    runner_plumbing(run_tasks_module, dataset)
    monkeypatch.setattr(
        run_tasks_module, "run_tasks_data_item", AsyncMock(return_value={"run_info": "ok"})
    )

    progress_calls = []

    async def _fake_log_progress(**kwargs):
        progress_calls.append(kwargs)

    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_progress", _fake_log_progress)

    async for _ in run_tasks_module.run_tasks(
        tasks=[],
        dataset_id=dataset.id,
        data=["a", "b", "c"],
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="cognify_pipeline",
    ):
        pass

    assert len(progress_calls) == 3
    assert all(call["total_items"] == 3 for call in progress_calls)
    # Every item counted exactly once, 1..N (order depends on gather scheduling,
    # not insertion order, so compare as a set).
    assert sorted(call["completed_items"] for call in progress_calls) == [1, 2, 3]


@pytest.mark.asyncio
async def test_run_tasks_progress_failure_does_not_break_the_run(monkeypatch, runner_plumbing):
    """Progress reporting must never fail the pipeline run (see run_tasks.py)."""
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    runner_plumbing(run_tasks_module, dataset)
    monkeypatch.setattr(
        run_tasks_module, "run_tasks_data_item", AsyncMock(return_value={"run_info": "ok"})
    )

    async def _broken_log_progress(**kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_progress", _broken_log_progress)

    from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted

    events = [
        event
        async for event in run_tasks_module.run_tasks(
            tasks=[],
            dataset_id=dataset.id,
            data=["a"],
            user=SimpleNamespace(id=uuid4(), tenant_id=None),
            pipeline_name="cognify_pipeline",
        )
    ]

    assert any(isinstance(e, PipelineRunCompleted) for e in events)


@pytest.mark.asyncio
async def test_run_tasks_counts_items_that_hard_raise(monkeypatch, runner_plumbing):
    """An item whose run_tasks_data_item call raises (rather than returning a
    PipelineRunErrored result dict — e.g. incremental loading with the default
    RAISE_INCREMENTAL_LOADING_ERRORS=true) must still be counted as "done" —
    _run_item's finally block, not a line after a plain return, is what
    guarantees this."""
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    runner_plumbing(run_tasks_module, dataset)

    async def _item_run(data_item, *_args, **_kwargs):
        if data_item == "b":
            raise RuntimeError("hard failure")
        return {"run_info": "ok"}

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", _item_run)

    progress_calls = []

    async def _fake_log_progress(**kwargs):
        progress_calls.append(kwargs)

    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_progress", _fake_log_progress)

    with pytest.raises(RuntimeError):
        async for _ in run_tasks_module.run_tasks(
            tasks=[],
            dataset_id=dataset.id,
            data=["a", "b", "c"],
            user=SimpleNamespace(id=uuid4(), tenant_id=None),
            pipeline_name="cognify_pipeline",
        ):
            pass

    # All three items counted, including the one that raised.
    assert len(progress_calls) == 3
    assert sorted(call["completed_items"] for call in progress_calls) == [1, 2, 3]


@pytest.mark.asyncio
async def test_run_tasks_throttles_progress_db_writes_on_large_batches(
    monkeypatch, runner_plumbing
):
    """100 items must not mean 100 progress inserts — only the first, the
    last, and ~20 in between (see progress_log_every in run_tasks.py), so a
    large batch doesn't add write pressure proportional to its size on the
    default SQLite relational backend."""
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    runner_plumbing(run_tasks_module, dataset)
    monkeypatch.setattr(
        run_tasks_module, "run_tasks_data_item", AsyncMock(return_value={"run_info": "ok"})
    )

    progress_calls = []

    async def _fake_log_progress(**kwargs):
        progress_calls.append(kwargs)

    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_progress", _fake_log_progress)

    data = [f"item-{i}" for i in range(100)]
    async for _ in run_tasks_module.run_tasks(
        tasks=[],
        dataset_id=dataset.id,
        data=data,
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="cognify_pipeline",
    ):
        pass

    # ~20 ticks, not 100 — and the very last item (100/100) is always among them.
    assert 15 <= len(progress_calls) <= 25
    assert any(call["completed_items"] == 100 for call in progress_calls)
    assert any(call["completed_items"] == 1 for call in progress_calls)


@pytest.mark.asyncio
async def test_log_pipeline_run_progress_updates_in_place():
    """Multiple progress ticks for the same run must update one row, not
    insert one per tick — otherwise pipeline_runs grows without bound as
    batches accumulate (the table's whole lifetime, not just one run)."""
    await create_db_and_tables()

    dataset_id = uuid4()
    pipeline_id = uuid4()

    pipeline_run = await log_pipeline_run_start(pipeline_id, "cognify_pipeline", dataset_id, None)
    pipeline_run_id = pipeline_run.pipeline_run_id

    for tick in range(1, 6):
        await log_pipeline_run_progress(
            pipeline_run_id=pipeline_run_id,
            pipeline_id=pipeline_id,
            pipeline_name="cognify_pipeline",
            dataset_id=dataset_id,
            completed_items=tick,
            total_items=5,
            current_stage=f"stage-{tick}",
        )

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(PipelineRun).filter(PipelineRun.pipeline_run_id == pipeline_run_id)
                )
            )
            .scalars()
            .all()
        )

    # log_pipeline_run_start's row + zero extra rows from the five ticks
    # that followed — not six rows.
    assert len(rows) == 1
    assert rows[0].run_info["progress"] == {
        "completed_items": 5,
        "total_items": 5,
        "current_stage": "stage-5",
    }


@pytest.mark.asyncio
async def test_log_pipeline_run_progress_drops_tick_after_terminal_status():
    """A tick that arrives with no STARTED row left for this pipeline_run_id
    but a terminal (COMPLETED) row already present. Today's only caller
    (run_tasks.py) can't produce this — the STARTED row is never deleted,
    so log_pipeline_run_progress's own first query always finds and updates
    it in place instead, terminal status or not. This test simulates a
    hypothetical future where STARTED rows get pruned/archived, to prove the
    fallback path won't insert a spurious new STARTED row with a created_at
    newer than the terminal row (which would make get_pipeline_status
    report STARTED for a completed run)."""
    await create_db_and_tables()

    dataset_id = uuid4()
    pipeline_id = uuid4()

    pipeline_run = await log_pipeline_run_start(pipeline_id, "cognify_pipeline", dataset_id, None)
    pipeline_run_id = pipeline_run.pipeline_run_id
    await log_pipeline_run_complete(
        pipeline_run_id, pipeline_id, "cognify_pipeline", dataset_id, None
    )

    # Simulate the STARTED row no longer existing (e.g. pruned/archived),
    # leaving only the terminal COMPLETED row — the scenario the fallback
    # insert's terminal-status guard exists for.
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        started_row = (
            await session.execute(
                select(PipelineRun)
                .filter(PipelineRun.pipeline_run_id == pipeline_run_id)
                .filter(PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_STARTED)
            )
        ).scalar_one()
        await session.delete(started_row)
        await session.commit()

    result = await log_pipeline_run_progress(
        pipeline_run_id=pipeline_run_id,
        pipeline_id=pipeline_id,
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        completed_items=1,
        total_items=1,
    )

    # The tick was dropped: no new row inserted, and the terminal row's
    # own status/run_info come back untouched.
    assert result.status.value == "DATASET_PROCESSING_COMPLETED"

    async with db_engine.get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(PipelineRun).filter(PipelineRun.pipeline_run_id == pipeline_run_id)
                )
            )
            .scalars()
            .all()
        )

    # Only the terminal row remains — no new STARTED row from the dropped tick.
    assert len(rows) == 1
    assert rows[0].status.value == "DATASET_PROCESSING_COMPLETED"


def test_queue_round_trips_pipeline_run_progress():
    pipeline_run_id = uuid4()
    progress = PipelineRunProgress(
        pipeline_run_id=pipeline_run_id,
        dataset_id=uuid4(),
        dataset_name="ds",
        completed_items=2,
        total_items=5,
        current_stage="extract_graph_from_data",
        stage_index=3,
        stage_total=4,
    )

    try:
        queues_module.push_to_queue(pipeline_run_id, progress)
        received = queues_module.get_from_queue(pipeline_run_id)

        assert received is progress
        assert received.status == "PipelineRunProgress"
        assert received.completed_items == 2
        assert received.total_items == 5
        assert received.current_stage == "extract_graph_from_data"
    finally:
        queues_module.remove_queue(pipeline_run_id)
