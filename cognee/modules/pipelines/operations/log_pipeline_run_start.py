from uuid import UUID
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from typing import Any

from cognee.modules.pipelines.utils import generate_pipeline_run_id, summarize_run_info_data


async def log_pipeline_run_start(
    pipeline_id: UUID, pipeline_name: str, dataset_id: UUID, data: Any
):
    data_info = summarize_run_info_data(data)

    pipeline_run_id = generate_pipeline_run_id(pipeline_id, dataset_id)

    pipeline_run = PipelineRun(
        pipeline_run_id=pipeline_run_id,
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        dataset_id=dataset_id,
        run_info={
            "data": data_info,
        },
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
