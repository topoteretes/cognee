"""Follow-up to SDK-591: read-time ABANDONED status must reach the dataset
status endpoints without corrupting the raw-status control flow they share
code with.

get_effective_pipeline_status() (SDK-591) computes ABANDONED for a stale
STARTED row, but is display-only. check_pipeline_run_qualification decides
whether a pipeline is already running or already done by comparing
get_pipeline_status()'s result against PipelineRunStatus members — feeding
it a string like "ABANDONED" (or "DATASET_PROCESSING_STARTED" as a bare
str instead of the enum) would make both comparisons permanently False,
silently letting a pipeline relaunch on top of one that is only reporting
stale, not actually gone.

Covered here, against a real relational DB (not a mock), because the bug
this guards against is specifically about which callable a given code path
imports:
- get_pipeline_status keeps returning raw PipelineRunStatus enum members
  for a stale STARTED row, not "ABANDONED".
- get_effective_pipeline_status_by_datasets / _progress_by_datasets (the
  new reporting siblings) report ABANDONED for the same row.
- check_pipeline_run_qualification still dedupes a stale-but-technically-
  running row as PipelineRunStarted — the regression the CI bot's suggested
  one-line fix would have introduced.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.infrastructure.databases.relational import (
    create_db_and_tables,
    get_relational_engine,
)
from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.layers.check_pipeline_run_qualification import (
    check_pipeline_run_qualification,
)
from cognee.modules.pipelines.methods.get_effective_pipeline_status import (
    EffectivePipelineRunStatus,
)
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunStarted
from cognee.modules.pipelines.operations.get_pipeline_status import (
    get_effective_pipeline_progress_by_datasets,
    get_effective_pipeline_status_by_datasets,
    get_pipeline_status,
)
from cognee.modules.pipelines.operations.log_pipeline_run_start import log_pipeline_run_start


async def _make_stale_started_run(dataset_id, pipeline_name, abandon_after_seconds=60):
    """A STARTED pipeline_runs row old enough to read as ABANDONED."""
    pipeline_run = await log_pipeline_run_start(uuid4(), pipeline_name, dataset_id, None)

    stale_created_at = datetime.now(timezone.utc) - timedelta(seconds=abandon_after_seconds * 2)
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        row = (
            await session.execute(
                select(PipelineRun).filter(
                    PipelineRun.pipeline_run_id == pipeline_run.pipeline_run_id
                )
            )
        ).scalar_one()
        row.created_at = stale_created_at
        await session.commit()

    return pipeline_run


@pytest.mark.asyncio
async def test_get_pipeline_status_returns_raw_enum_for_a_stale_row(monkeypatch):
    """The control-flow lookup must never see "ABANDONED" — a stale STARTED
    row still reads as the raw PipelineRunStatus.DATASET_PROCESSING_STARTED
    enum member."""
    monkeypatch.setenv("PIPELINE_RUN_ABANDON_AFTER_SECONDS", "60")
    await create_db_and_tables()
    dataset_id = uuid4()
    pipeline_name = "cognify_pipeline"
    await _make_stale_started_run(dataset_id, pipeline_name)

    statuses = await get_pipeline_status([dataset_id], pipeline_name)

    assert statuses[str(dataset_id)] is PipelineRunStatus.DATASET_PROCESSING_STARTED


@pytest.mark.asyncio
async def test_get_effective_pipeline_status_by_datasets_reports_abandoned(monkeypatch):
    monkeypatch.setenv("PIPELINE_RUN_ABANDON_AFTER_SECONDS", "60")
    await create_db_and_tables()
    dataset_id = uuid4()
    pipeline_name = "cognify_pipeline"
    await _make_stale_started_run(dataset_id, pipeline_name)

    statuses = await get_effective_pipeline_status_by_datasets([dataset_id], pipeline_name)

    assert statuses[str(dataset_id)] == EffectivePipelineRunStatus.ABANDONED


@pytest.mark.asyncio
async def test_get_effective_pipeline_status_by_datasets_leaves_a_fresh_row_alone(monkeypatch):
    monkeypatch.setenv("PIPELINE_RUN_ABANDON_AFTER_SECONDS", "1800")
    await create_db_and_tables()
    dataset_id = uuid4()
    pipeline_name = "cognify_pipeline"
    await log_pipeline_run_start(uuid4(), pipeline_name, dataset_id, None)

    statuses = await get_effective_pipeline_status_by_datasets([dataset_id], pipeline_name)

    assert statuses[str(dataset_id)] == EffectivePipelineRunStatus.DATASET_PROCESSING_STARTED


@pytest.mark.asyncio
async def test_get_effective_pipeline_progress_by_datasets_reports_abandoned(monkeypatch):
    """{status, progress} shape: status flips to ABANDONED, progress is
    passed through unchanged (there is none yet for a freshly-started run)."""
    monkeypatch.setenv("PIPELINE_RUN_ABANDON_AFTER_SECONDS", "60")
    await create_db_and_tables()
    dataset_id = uuid4()
    pipeline_name = "cognify_pipeline"
    await _make_stale_started_run(dataset_id, pipeline_name)

    progress = await get_effective_pipeline_progress_by_datasets([dataset_id], pipeline_name)

    assert progress[str(dataset_id)]["status"] == EffectivePipelineRunStatus.ABANDONED
    assert progress[str(dataset_id)]["progress"] is None


@pytest.mark.asyncio
async def test_check_pipeline_run_qualification_dedupes_a_stale_row_as_running(monkeypatch):
    """This is the regression test for the bug the CI bot's suggested fix
    (applying get_effective_pipeline_status inside get_pipeline_status)
    would have introduced: a stale STARTED row reports ABANDONED to a
    human reading /status, but a second pipeline launch against the same
    dataset must still be blocked, because as far as the pipeline runner is
    concerned the row could still be a live worker."""
    monkeypatch.setenv("PIPELINE_RUN_ABANDON_AFTER_SECONDS", "60")
    await create_db_and_tables()
    dataset_id = uuid4()
    pipeline_name = "cognify_pipeline"
    await _make_stale_started_run(dataset_id, pipeline_name)

    # Sanity: this row does read as ABANDONED through the reporting path.
    effective = await get_effective_pipeline_status_by_datasets([dataset_id], pipeline_name)
    assert effective[str(dataset_id)] == EffectivePipelineRunStatus.ABANDONED

    dataset = Dataset(id=dataset_id, name="ds", owner_id=uuid4())

    result = await check_pipeline_run_qualification(dataset, [], pipeline_name)

    assert isinstance(result, PipelineRunStarted)
    assert result.dataset_id == dataset_id
