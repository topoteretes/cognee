import asyncio
from uuid import UUID
from typing import Dict, Optional, Union

from cognee.modules.pipelines.layers.setup_and_check_environment import (
    setup_and_check_environment,
)

from cognee.shared.logging_utils import get_logger
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.layers import validate_pipeline_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.pipelines.layers.check_pipeline_run_qualification import (
    check_pipeline_run_qualification,
)
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunStarted,
)
from typing import Any

logger = get_logger("cognee.pipeline")

update_status_lock = asyncio.Lock()


async def run_pipeline(
    tasks: list[Task],
    data=None,
    datasets: Optional[Union[str, list[str], list[UUID]]] = None,
    user: Optional[User] = None,
    pipeline_name: str = "custom_pipeline",
    use_pipeline_cache: bool = False,
    vector_db_config: Optional[dict] = None,
    graph_db_config: Optional[dict] = None,
    incremental_loading: bool = False,
    context: Optional[Dict] = None,
    data_per_batch: int = 20,
):
    validate_pipeline_tasks(tasks)
    await setup_and_check_environment(vector_db_config, graph_db_config)

    user, authorized_datasets = await resolve_authorized_user_datasets(datasets, user)

    for dataset in authorized_datasets:
        async for run_info in run_pipeline_per_dataset(
            dataset=dataset,
            user=user,
            tasks=tasks,
            data=data,
            pipeline_name=pipeline_name,
            context={"dataset": dataset, "pipeline_name": pipeline_name},
            use_pipeline_cache=use_pipeline_cache,
            incremental_loading=incremental_loading,
            data_per_batch=data_per_batch,
        ):
            yield run_info


async def run_pipeline_per_dataset(
    dataset: Dataset,
    user: User,
    tasks: list[Task],
    data: Optional[list[Data]] = None,
    pipeline_name: str = "custom_pipeline",
    use_pipeline_cache=False,
    incremental_loading=False,
    context: Optional[Dict] = None,
    data_per_batch: int = 20,
):
    # Will only be used if ENABLE_BACKEND_ACCESS_CONTROL is set to True
    await set_database_global_context_variables(dataset.id, dataset.owner_id)

    if not data:
        data = await get_dataset_data(dataset_id=dataset.id)

    process_pipeline_status = await check_pipeline_run_qualification(dataset, data, pipeline_name)
    if process_pipeline_status:
        # If pipeline was already processed or is currently being processed
        # return status information to async generator and finish execution
        if use_pipeline_cache:
            # If pipeline caching is enabled we do not proceed with re-processing
            yield process_pipeline_status
            return
        else:
            # If pipeline caching is disabled we always return pipeline started information and proceed with re-processing
            yield PipelineRunStarted(
                pipeline_run_id=process_pipeline_status.pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                payload=data,
            )

    pipeline_run = run_tasks(
        tasks,
        dataset.id,
        data,
        user,
        pipeline_name,
        context,
        incremental_loading,
        data_per_batch,
    )

    async for pipeline_run_info in pipeline_run:
        yield pipeline_run_info
