"""The ``run_info`` column of a run lists the ids the run processed, never the corpus (COG-6089).

It used to list every data item of the dataset in the STARTED row and again in the COMPLETED or
ERRORED row, about 200 KB per row for 5000 items. Every row now carries ``data`` (the ids
completed in the run) and ``errored_data`` (the ids that failed in it), empty on STARTED, and
the writers take the ids from the caller instead of the run's input.
"""

from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.infrastructure.databases.relational import create_db_and_tables, get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.operations.log_pipeline_run_complete import (
    log_pipeline_run_complete,
)
from cognee.modules.pipelines.operations.log_pipeline_run_error import log_pipeline_run_error
from cognee.modules.pipelines.operations.log_pipeline_run_start import log_pipeline_run_start

# The recovery lock and progress go on the STARTED row, the scrubbed error on the ERRORED row.
RUN_INFO_KEYS = {"data", "errored_data", "recovery_lock", "progress", "error"}


async def _rows(dataset_id):
    async with get_relational_engine().get_async_session() as session:
        rows = await session.execute(
            select(PipelineRun).filter(PipelineRun.dataset_id == dataset_id)
        )
    return {row.status: row for row in rows.scalars().all()}


@pytest.mark.asyncio
async def test_a_completed_run_lists_the_ids_it_completed():
    await create_db_and_tables()
    pipeline_id, dataset_id, done = uuid4(), uuid4(), str(uuid4())

    run = await log_pipeline_run_start(
        pipeline_id, "add_pipeline", dataset_id, recovery_token="lock"
    )
    await log_pipeline_run_complete(
        run.pipeline_run_id, pipeline_id, "add_pipeline", dataset_id, completed_data_ids=[done]
    )

    rows = await _rows(dataset_id)
    started = rows[PipelineRunStatus.DATASET_PROCESSING_STARTED].run_info
    completed = rows[PipelineRunStatus.DATASET_PROCESSING_COMPLETED].run_info
    assert started == {"data": [], "errored_data": [], "recovery_lock": "lock"}
    assert completed == {"data": [done], "errored_data": []}


@pytest.mark.asyncio
async def test_an_errored_run_lists_what_completed_and_what_failed():
    await create_db_and_tables()
    pipeline_id, dataset_id, done, failed = uuid4(), uuid4(), str(uuid4()), str(uuid4())

    run = await log_pipeline_run_start(pipeline_id, "cognify_pipeline", dataset_id)
    await log_pipeline_run_error(
        run.pipeline_run_id,
        pipeline_id,
        "cognify_pipeline",
        dataset_id,
        ValueError("boom"),
        completed_data_ids=[done],
        errored_data_ids=[failed],
    )

    rows = await _rows(dataset_id)
    errored = rows[PipelineRunStatus.DATASET_PROCESSING_ERRORED].run_info
    assert errored["data"] == [done]
    assert errored["errored_data"] == [failed]
    assert all(set(row.run_info) <= RUN_INFO_KEYS for row in rows.values())


@pytest.mark.asyncio
async def test_a_run_closed_from_outside_records_no_ids():
    """Startup recovery closes a run whose process is gone; it knows no ids."""
    await create_db_and_tables()
    pipeline_id, dataset_id = uuid4(), uuid4()

    run = await log_pipeline_run_start(pipeline_id, "cognify_pipeline", dataset_id)
    await log_pipeline_run_error(
        run.pipeline_run_id, pipeline_id, "cognify_pipeline", dataset_id, ValueError("abandoned")
    )

    errored = (await _rows(dataset_id))[PipelineRunStatus.DATASET_PROCESSING_ERRORED].run_info
    assert errored["data"] == [] and errored["errored_data"] == []
