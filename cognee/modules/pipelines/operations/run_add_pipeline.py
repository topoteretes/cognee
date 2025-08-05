from typing import Optional
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.operations import log_pipeline_run_initiated
from cognee.modules.pipelines.utils import generate_pipeline_id, validate_pipeline_inputs
from cognee.context_global_variables import set_database_global_context_variables

logger = get_logger("add.pipeline")


@validate_pipeline_inputs
async def run_add_pipeline(
    tasks: list[Task],
    data,
    dataset: Dataset,
    user: User,
    pipeline_name: str = "add_pipeline",
    incremental_loading: Optional[bool] = True,
):
    await set_database_global_context_variables(dataset.id, dataset.owner_id)

    pipeline_run = run_tasks(
        tasks,
        dataset.id,
        data,
        user,
        pipeline_name,
        {
            "user": user,
            "dataset": dataset,
        },
        incremental_loading,
    )

    async for pipeline_run_info in pipeline_run:
        yield pipeline_run_info

    pipeline_id = generate_pipeline_id(user.id, dataset.id, pipeline_name)

    # Refresh the cognify pipeline status after we add new files.
    # Without this the cognify_pipeline status will be DATASET_PROCESSING_COMPLETED and will skip the execution.
    await log_pipeline_run_initiated(
        pipeline_id=pipeline_id,
        pipeline_name="cognify_pipeline",
        dataset_id=dataset.id,
    )
