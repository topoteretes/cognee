"""A terminal row (COMPLETED or ERRORED) is written once per pipeline_run_id,
never twice (SDK-591 review, github.com/topoteretes/cognee/pull/4983#discussion_r4004955273).

pipeline_runs is append-only, so two terminal writers firing for the same run
is not hypothetical: startup recovery closes a run as ERRORED under
`AbandonedPipelineRunError` when it believes the process that started it is
gone, but the age floor next to the origin check is a "practically never",
not a "never" — and a process that outlives the guess reaches its own
completion path afterwards. Without a guard that second write lands as
COMPLETED on top of a graph the recovery rollback already deleted, which is
worse than the stale status this was built to fix. Both terminal writers
check for an existing terminal row first and skip the second write instead.
"""

from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.infrastructure.databases.relational import (
    create_db_and_tables,
    get_relational_engine,
)
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.operations.log_pipeline_run_complete import (
    log_pipeline_run_complete,
)
from cognee.modules.pipelines.operations.log_pipeline_run_error import log_pipeline_run_error
from cognee.modules.pipelines.operations.log_pipeline_run_start import log_pipeline_run_start


async def _rows_for(pipeline_run_id):
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        return (
            (
                await session.execute(
                    select(PipelineRun).filter(PipelineRun.pipeline_run_id == pipeline_run_id)
                )
            )
            .scalars()
            .all()
        )


@pytest.mark.asyncio
async def test_complete_after_error_does_not_overwrite_the_error():
    """The exact race: recovery closes the run as ERRORED, then the process
    it thought was dead reaches its own success path. COMPLETED must not
    land on top."""
    await create_db_and_tables()

    dataset_id = uuid4()
    pipeline_id = uuid4()

    pipeline_run = await log_pipeline_run_start(pipeline_id, "cognify_pipeline", dataset_id, None)
    pipeline_run_id = pipeline_run.pipeline_run_id

    await log_pipeline_run_error(
        pipeline_run_id, pipeline_id, "cognify_pipeline", dataset_id, None, RuntimeError("dead")
    )

    result = await log_pipeline_run_complete(
        pipeline_run_id, pipeline_id, "cognify_pipeline", dataset_id, None
    )

    rows = await _rows_for(pipeline_run_id)
    terminal_rows = [
        r
        for r in rows
        if r.status
        in (
            PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
            PipelineRunStatus.DATASET_PROCESSING_ERRORED,
        )
    ]
    assert len(terminal_rows) == 1
    assert terminal_rows[0].status == PipelineRunStatus.DATASET_PROCESSING_ERRORED
    # The skipped writer hands back the row that actually won, not None —
    # a caller checking the outcome sees the true terminal state.
    assert result.status == PipelineRunStatus.DATASET_PROCESSING_ERRORED


@pytest.mark.asyncio
async def test_error_after_complete_does_not_overwrite_the_completion():
    """The mirror case: a late failure path (or a second recovery sweep)
    reaching log_pipeline_run_error after the run already completed."""
    await create_db_and_tables()

    dataset_id = uuid4()
    pipeline_id = uuid4()

    pipeline_run = await log_pipeline_run_start(pipeline_id, "cognify_pipeline", dataset_id, None)
    pipeline_run_id = pipeline_run.pipeline_run_id

    await log_pipeline_run_complete(
        pipeline_run_id, pipeline_id, "cognify_pipeline", dataset_id, None
    )

    result = await log_pipeline_run_error(
        pipeline_run_id, pipeline_id, "cognify_pipeline", dataset_id, None, RuntimeError("late")
    )

    rows = await _rows_for(pipeline_run_id)
    terminal_rows = [
        r
        for r in rows
        if r.status
        in (
            PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
            PipelineRunStatus.DATASET_PROCESSING_ERRORED,
        )
    ]
    assert len(terminal_rows) == 1
    assert terminal_rows[0].status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED
    assert result.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED


@pytest.mark.asyncio
async def test_complete_after_complete_is_a_no_op_not_a_second_row():
    """Not just cross-status: the same writer firing twice for one run
    (a retried caller, a duplicated event) must not double-insert either."""
    await create_db_and_tables()

    dataset_id = uuid4()
    pipeline_id = uuid4()

    pipeline_run = await log_pipeline_run_start(pipeline_id, "cognify_pipeline", dataset_id, None)
    pipeline_run_id = pipeline_run.pipeline_run_id

    await log_pipeline_run_complete(
        pipeline_run_id, pipeline_id, "cognify_pipeline", dataset_id, None
    )
    await log_pipeline_run_complete(
        pipeline_run_id, pipeline_id, "cognify_pipeline", dataset_id, None
    )

    rows = await _rows_for(pipeline_run_id)
    completed_rows = [r for r in rows if r.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED]
    assert len(completed_rows) == 1


@pytest.mark.asyncio
async def test_the_first_terminal_write_for_a_run_still_lands():
    """The guard must not block the ordinary, non-racing case: the first
    terminal write for a run with no prior terminal row goes through."""
    await create_db_and_tables()

    dataset_id = uuid4()
    pipeline_id = uuid4()

    pipeline_run = await log_pipeline_run_start(pipeline_id, "cognify_pipeline", dataset_id, None)
    pipeline_run_id = pipeline_run.pipeline_run_id

    result = await log_pipeline_run_complete(
        pipeline_run_id, pipeline_id, "cognify_pipeline", dataset_id, None
    )

    assert result.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED
    rows = await _rows_for(pipeline_run_id)
    assert len(rows) == 2  # the STARTED row plus this COMPLETED row
