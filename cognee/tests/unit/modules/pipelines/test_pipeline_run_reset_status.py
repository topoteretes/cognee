"""Reset-status bookkeeping in pipeline_runs (SDK-595).

A reset says "this pipeline may run again". It used to say so by inserting an
INITIATED row under a freshly minted pipeline_run_id, which no STARTED or
terminal row ever shared, so the row stayed in pipeline_runs forever
describing work that had already finished. add() resets two pipelines on
every call, so an ordinary workspace grew one or two of these per add.

The marker is now control state with an owner: it carries the identity of the
run it resets, and log_pipeline_run_start clears it when the run it unblocked
begins.

Covered here:
- The reset row carries the id of the run it resets, not a new one.
- Resetting leaves the run's terminal row intact (the COMPLETED-row
  existence checks in edge evidence and graph warmup depend on it).
- Two runs on one dataset, with a reset between them the way add() does it,
  leave no row that a later row does not supersede.
- The whole add() shape: reset_dataset_pipeline_run_status resets both
  add_pipeline and cognify_pipeline, and each marker is cleared by its own
  pipeline's next start.
- A reset by someone other than the user who runs the pipeline is still
  cleared. This is what startup recovery does, acting as the dataset owner.
- The status a reset produces is still INITIATED, through both
  get_pipeline_status and get_pipeline_progress, so
  check_pipeline_run_qualification does not skip the re-run.
- The one marker an add() leaves behind when no cognify follows, pinned at
  exactly one row so the accepted limit is visible if it ever changes.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.infrastructure.databases.relational import (
    create_db_and_tables,
    get_relational_engine,
)
from cognee.modules.pipelines.layers.reset_dataset_pipeline_run_status import (
    reset_dataset_pipeline_run_status,
)
from cognee.modules.pipelines.methods import reset_pipeline_run_status
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import (
    get_pipeline_progress,
    get_pipeline_status,
)
from cognee.modules.pipelines.operations.log_pipeline_run_complete import (
    log_pipeline_run_complete,
)
from cognee.modules.pipelines.operations.log_pipeline_run_start import log_pipeline_run_start
from cognee.modules.pipelines.utils.generate_pipeline_id import generate_pipeline_id

PIPELINE_NAME = "cognify_pipeline"


async def _rows_for_dataset(dataset_id, pipeline_name=None):
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        query = select(PipelineRun).filter(PipelineRun.dataset_id == dataset_id)
        if pipeline_name is not None:
            query = query.filter(PipelineRun.pipeline_name == pipeline_name)
        return (await session.execute(query.order_by(PipelineRun.created_at))).scalars().all()


async def _run_once(dataset_id, user_id, pipeline_name=PIPELINE_NAME):
    """One pipeline run the way run_tasks does it: pipeline_id derived from
    the acting user, a STARTED row, then a terminal row under the same id."""
    pipeline_id = generate_pipeline_id(user_id, dataset_id, pipeline_name)
    run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset_id, None)
    await log_pipeline_run_complete(
        run.pipeline_run_id, pipeline_id, pipeline_name, dataset_id, None
    )
    return run.pipeline_run_id


async def _latest_run(dataset_id, pipeline_name=PIPELINE_NAME):
    rows = await _rows_for_dataset(dataset_id, pipeline_name)
    return rows[-1]


@pytest.mark.asyncio
async def test_reset_reuses_the_run_id_and_keeps_the_terminal_row():
    await create_db_and_tables()

    user_id, dataset_id = uuid4(), uuid4()
    # Reset as somebody other than the runner, so a marker that took its
    # identity from the acting user instead of from the run is visible.
    resetting_user_id = uuid4()

    pipeline_run_id = await _run_once(dataset_id, user_id)
    await reset_pipeline_run_status(await _latest_run(dataset_id), user_id=resetting_user_id)

    rows = await _rows_for_dataset(dataset_id)

    # Every row belongs to the one run: no third identity was invented.
    assert {row.pipeline_run_id for row in rows} == {pipeline_run_id}
    assert {row.pipeline_id for row in rows} == {
        generate_pipeline_id(user_id, dataset_id, PIPELINE_NAME)
    }
    # ... while who asked for the reset is recorded as what it is.
    assert rows[-1].user_id == resetting_user_id
    assert [row.status for row in rows] == [
        PipelineRunStatus.DATASET_PROCESSING_STARTED,
        PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        PipelineRunStatus.DATASET_PROCESSING_INITIATED,
    ]


@pytest.mark.asyncio
async def test_reset_status_is_initiated_so_the_rerun_is_not_skipped():
    """The reset exists so a COMPLETED run does not skip execution."""
    await create_db_and_tables()

    user_id, dataset_id = uuid4(), uuid4()

    await _run_once(dataset_id, user_id)
    assert (await get_pipeline_status([dataset_id], PIPELINE_NAME))[str(dataset_id)] == (
        PipelineRunStatus.DATASET_PROCESSING_COMPLETED
    )

    await reset_pipeline_run_status(await _latest_run(dataset_id), user_id=user_id)

    assert (await get_pipeline_status([dataset_id], PIPELINE_NAME))[str(dataset_id)] == (
        PipelineRunStatus.DATASET_PROCESSING_INITIATED
    )
    # The progress reader shares the lookup and must agree with it.
    progress = (await get_pipeline_progress([dataset_id], PIPELINE_NAME))[str(dataset_id)]
    assert progress["status"] == PipelineRunStatus.DATASET_PROCESSING_INITIATED
    assert progress["progress"] is None


@pytest.mark.asyncio
async def test_repeated_runs_leave_no_unsuperseded_row():
    await create_db_and_tables()

    user_id, dataset_id = uuid4(), uuid4()

    first_run_id = await _run_once(dataset_id, user_id)
    await reset_pipeline_run_status(await _latest_run(dataset_id), user_id=user_id)
    second_run_id = await _run_once(dataset_id, user_id)

    rows = await _rows_for_dataset(dataset_id)

    # Two runs, two rows each (STARTED then COMPLETED). The reset marker that
    # unblocked the second run was cleared when that run started, so nothing
    # is left at INITIATED and neither run reads as queued.
    assert len(rows) == 4
    assert not [row for row in rows if row.status is PipelineRunStatus.DATASET_PROCESSING_INITIATED]
    per_run = {}
    for row in rows:
        per_run.setdefault(row.pipeline_run_id, []).append(row.status)
    assert per_run == {
        first_run_id: [
            PipelineRunStatus.DATASET_PROCESSING_STARTED,
            PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        ],
        second_run_id: [
            PipelineRunStatus.DATASET_PROCESSING_STARTED,
            PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        ],
    }


@pytest.mark.asyncio
async def test_add_shaped_reset_clears_both_pipelines_markers():
    """add() resets add_pipeline and cognify_pipeline together, through the
    layer both add() and forget() call. Each marker is cleared by its own
    pipeline's next start, and neither clears the other's."""
    await create_db_and_tables()

    user = SimpleNamespace(id=uuid4())
    dataset_id = uuid4()

    await _run_once(dataset_id, user.id, "add_pipeline")
    await _run_once(dataset_id, user.id, "cognify_pipeline")

    await reset_dataset_pipeline_run_status(
        dataset_id, user, pipeline_names=["add_pipeline", "cognify_pipeline"]
    )

    initiated = [
        row
        for row in await _rows_for_dataset(dataset_id)
        if row.status is PipelineRunStatus.DATASET_PROCESSING_INITIATED
    ]
    assert sorted(row.pipeline_name for row in initiated) == ["add_pipeline", "cognify_pipeline"]

    await _run_once(dataset_id, user.id, "add_pipeline")
    still_initiated = [
        row
        for row in await _rows_for_dataset(dataset_id)
        if row.status is PipelineRunStatus.DATASET_PROCESSING_INITIATED
    ]
    # The add run cleared its own marker and left cognify's alone.
    assert [row.pipeline_name for row in still_initiated] == ["cognify_pipeline"]

    await _run_once(dataset_id, user.id, "cognify_pipeline")
    assert not [
        row
        for row in await _rows_for_dataset(dataset_id)
        if row.status is PipelineRunStatus.DATASET_PROCESSING_INITIATED
    ]


@pytest.mark.asyncio
async def test_marker_is_cleared_when_the_resetter_is_not_the_runner():
    """Startup recovery resets as the dataset owner, and a shared dataset can
    be cognified by someone who does not own it. The marker belongs to the
    dataset and pipeline, not to whoever happened to write it, so the next run
    clears it whichever user starts it."""
    await create_db_and_tables()

    owner_id, other_user_id, dataset_id = uuid4(), uuid4(), uuid4()

    await _run_once(dataset_id, owner_id)
    await reset_pipeline_run_status(await _latest_run(dataset_id), user_id=owner_id)
    await _run_once(dataset_id, other_user_id)

    assert not [
        row
        for row in await _rows_for_dataset(dataset_id)
        if row.status is PipelineRunStatus.DATASET_PROCESSING_INITIATED
    ]


@pytest.mark.asyncio
async def test_add_without_cognify_leaves_exactly_one_marker():
    """add() resets cognify_pipeline but only runs add_pipeline, so until a
    cognify follows, one cognify marker stands. That is the truthful reading
    of the state (data is staged and uncognified) and it is bounded: repeated
    adds do not stack markers, because the reset skips a pipeline already at
    INITIATED. This pins the bound, which is the part worth noticing if it
    ever changes."""
    await create_db_and_tables()

    user = SimpleNamespace(id=uuid4())
    dataset_id = uuid4()

    await _run_once(dataset_id, user.id, "add_pipeline")
    await _run_once(dataset_id, user.id, "cognify_pipeline")

    for _ in range(3):
        await reset_dataset_pipeline_run_status(
            dataset_id, user, pipeline_names=["add_pipeline", "cognify_pipeline"]
        )
        await _run_once(dataset_id, user.id, "add_pipeline")

    initiated = [
        row
        for row in await _rows_for_dataset(dataset_id)
        if row.status is PipelineRunStatus.DATASET_PROCESSING_INITIATED
    ]
    assert [row.pipeline_name for row in initiated] == ["cognify_pipeline"]
