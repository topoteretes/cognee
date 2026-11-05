"""The ``run_info`` column of a run holds only its own bookkeeping, never the data items (COG-6089).

It used to list every data item of the dataset in the STARTED row and again in the COMPLETED or
ERRORED row, about 200 KB per row for 5000 items.
"""

from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.infrastructure.databases.relational import create_db_and_tables, get_relational_engine
from cognee.modules.pipelines.models import PipelineRun
from cognee.modules.pipelines.operations.log_pipeline_run_complete import (
    log_pipeline_run_complete,
)
from cognee.modules.pipelines.operations.log_pipeline_run_error import log_pipeline_run_error
from cognee.modules.pipelines.operations.log_pipeline_run_start import log_pipeline_run_start

# The recovery lock and progress go on the STARTED row, the scrubbed error on the ERRORED row.
RUN_INFO_KEYS = {"recovery_lock", "progress", "error"}


@pytest.mark.asyncio
async def test_run_info_holds_only_run_bookkeeping():
    await create_db_and_tables()
    pipeline_id, dataset_id = uuid4(), uuid4()

    completed = await log_pipeline_run_start(
        pipeline_id, "add_pipeline", dataset_id, recovery_token="lock"
    )
    await log_pipeline_run_complete(
        completed.pipeline_run_id, pipeline_id, "add_pipeline", dataset_id
    )
    errored = await log_pipeline_run_start(pipeline_id, "add_pipeline", dataset_id)
    await log_pipeline_run_error(
        errored.pipeline_run_id, pipeline_id, "add_pipeline", dataset_id, ValueError("boom")
    )

    async with get_relational_engine().get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(PipelineRun).filter(PipelineRun.dataset_id == dataset_id)
                )
            )
            .scalars()
            .all()
        )

    assert len(rows) == 4
    assert all(set(row.run_info) <= RUN_INFO_KEYS for row in rows)
