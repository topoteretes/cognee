from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.methods import get_pipeline_run_by_dataset
from cognee.shared.logging_utils import get_logger

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunStarted,
)

logger = get_logger(__name__)


async def pipeline_status_check(dataset, data, pipeline_name):
    # async with update_status_lock: TODO: Add UI lock to prevent multiple backend requests
    if isinstance(dataset, Dataset):
        task_status = await get_pipeline_status([dataset.id], pipeline_name)
    else:
        task_status = [
            PipelineRunStatus.DATASET_PROCESSING_COMPLETED
        ]  # TODO: this is a random assignment, find permanent solution

    if str(dataset.id) in task_status:
        if task_status[str(dataset.id)] == PipelineRunStatus.DATASET_PROCESSING_STARTED:
            logger.info("Dataset %s is already being processed.", dataset.id)
            pipeline_run = await get_pipeline_run_by_dataset(dataset.id, pipeline_name)
            yield PipelineRunStarted(
                pipeline_run_id=pipeline_run.pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                payload=data,
            )
            return
        elif task_status[str(dataset.id)] == PipelineRunStatus.DATASET_PROCESSING_COMPLETED:
            logger.info("Dataset %s is already processed.", dataset.id)
            pipeline_run = await get_pipeline_run_by_dataset(dataset.id, pipeline_name)
            yield PipelineRunCompleted(
                pipeline_run_id=pipeline_run.pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
            )
            return
