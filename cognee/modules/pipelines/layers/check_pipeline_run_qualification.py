from typing import Union, Optional
from cognee.modules.data.models import Dataset
from cognee.modules.data.models import Data
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.methods import get_pipeline_run_by_dataset
from cognee.shared.logging_utils import get_logger

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunStarted,
)

logger = get_logger(__name__)


async def check_pipeline_run_qualification(
    dataset: Dataset, data: list[Data], pipeline_name: str
) -> Optional[Union[PipelineRunStarted, PipelineRunCompleted]]:
    """
    Function used to determine if pipeline is currently being processed or was already processed.
    In case pipeline was or is being processed return value is returned and current pipline execution should be stopped.
    In case pipeline is not or was not processed there will be no return value and pipeline processing can start.

    Args:
        dataset: Dataset object
        data: List of Data
        pipeline_name: pipeline name

    Returns: Pipeline state if it is being processed or was already processed

    """

    # async with update_status_lock: TODO: Add UI lock to prevent multiple backend requests
    if isinstance(dataset, Dataset):
        task_status = await get_pipeline_status([dataset.id], pipeline_name)
    else:
        task_status = {}

    if str(dataset.id) in task_status:
        if task_status[str(dataset.id)] == PipelineRunStatus.DATASET_PROCESSING_STARTED:
            logger.info("Dataset %s is already being processed.", dataset.id)
            pipeline_run = await get_pipeline_run_by_dataset(dataset.id, pipeline_name)
            return PipelineRunStarted(
                pipeline_run_id=pipeline_run.pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                payload=data,
            )
        elif task_status[str(dataset.id)] == PipelineRunStatus.DATASET_PROCESSING_COMPLETED:
            logger.info("Dataset %s is already processed.", dataset.id)
            pipeline_run = await get_pipeline_run_by_dataset(dataset.id, pipeline_name)
            return PipelineRunCompleted(
                pipeline_run_id=pipeline_run.pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
            )

    return
