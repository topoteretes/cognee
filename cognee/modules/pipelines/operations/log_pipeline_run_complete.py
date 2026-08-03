from uuid import UUID
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from typing import Any

from cognee.modules.pipelines.utils import summarize_run_info_data


async def log_pipeline_run_complete(
    pipeline_run_id: UUID,
    pipeline_id: UUID,
    pipeline_name: str,
    dataset_id: UUID,
    data: Any,
    run_info_extra: dict | None = None,
):
    data_info = summarize_run_info_data(data)

    run_info = {"data": data_info}
    # Extra keys let a completed-but-partial run record why it stopped early and how
    # much is left (e.g. stopped_reason="budget_exhausted", documents_remaining=N),
    # without adding a new PipelineRunStatus. run_info is a JSON column, no migration.
    if run_info_extra:
        run_info.update(run_info_extra)

    pipeline_run = PipelineRun(
        pipeline_run_id=pipeline_run_id,
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        dataset_id=dataset_id,
        run_info=run_info,
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
