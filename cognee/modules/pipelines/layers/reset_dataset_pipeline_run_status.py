from uuid import UUID
from typing import Optional, List

from cognee.modules.pipelines.methods import get_pipeline_runs_by_dataset, reset_pipeline_run_status
from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.modules.users.models import User


async def reset_dataset_pipeline_run_status(
    dataset_id: UUID, user: User, pipeline_names: Optional[list[str]] = None
):
    """Reset the status of all (or selected) pipeline runs for a dataset.

    If *pipeline_names* is given, only runs whose *pipeline_name* is in
    that list are touched.
    """
    related_pipeline_runs = await get_pipeline_runs_by_dataset(dataset_id)

    for pipeline_run in related_pipeline_runs:
        # Skip runs that are initiated
        if pipeline_run.status is PipelineRunStatus.DATASET_PROCESSING_INITIATED:
            continue

        # If a name filter is provided, skip non-matching runs
        if pipeline_names is not None and pipeline_run.pipeline_name not in pipeline_names:
            continue

        await reset_pipeline_run_status(user.id, dataset_id, pipeline_run.pipeline_name)
