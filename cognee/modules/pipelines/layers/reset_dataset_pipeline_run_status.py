from uuid import UUID
from cognee.modules.pipelines.methods import get_pipeline_runs_by_dataset, reset_pipeline_run_status
from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.modules.users.models import User


async def reset_dataset_pipeline_run_status(dataset_id: UUID, user: User):
    related_pipeline_runs = await get_pipeline_runs_by_dataset(dataset_id)

    for pipeline_run in related_pipeline_runs:
        if pipeline_run.status is not PipelineRunStatus.DATASET_PROCESSING_INITIATED:
            await reset_pipeline_run_status(user.id, dataset_id, pipeline_run.pipeline_name)
