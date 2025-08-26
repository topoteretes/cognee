from uuid import UUID
from cognee.modules.pipelines.utils.generate_pipeline_id import generate_pipeline_id
from cognee.modules.pipelines.operations.log_pipeline_run_initiated import (
    log_pipeline_run_initiated,
)


async def reset_pipeline_run_status(user_id: UUID, dataset_id: UUID, pipeline_name: str):
    pipeline_id = generate_pipeline_id(user_id, dataset_id, pipeline_name)

    # Without this the pipeline status will be DATASET_PROCESSING_COMPLETED and will skip the execution.
    await log_pipeline_run_initiated(
        pipeline_id=pipeline_id,
        pipeline_name=pipeline_name,
        dataset_id=dataset_id,
    )
