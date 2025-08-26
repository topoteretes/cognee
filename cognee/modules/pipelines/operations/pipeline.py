import asyncio
from uuid import UUID
from typing import Union

from cognee.modules.pipelines.layers.environment_setup_and_checks import (
    environment_setup_and_checks,
)
from cognee.shared.logging_utils import get_logger
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.operations.run_tasks import run_tasks

from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.pipelines.layers.process_pipeline_check import process_pipeline_check

logger = get_logger("cognee.pipeline")

update_status_lock = asyncio.Lock()


async def cognee_pipeline(
    tasks: list[Task],
    data=None,
    datasets: Union[str, list[str], list[UUID]] = None,
    user: User = None,
    pipeline_name: str = "custom_pipeline",
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    incremental_loading: bool = False,
):
    await environment_setup_and_checks(vector_db_config, graph_db_config)

    user, authorized_datasets = await resolve_authorized_user_datasets(datasets, user)

    for dataset in authorized_datasets:
        async for run_info in run_pipeline(
            dataset=dataset,
            user=user,
            tasks=tasks,
            data=data,
            pipeline_name=pipeline_name,
            context={"dataset": dataset},
            incremental_loading=incremental_loading,
        ):
            yield run_info


async def run_pipeline(
    dataset: Dataset,
    user: User,
    tasks: list[Task],
    data=None,
    pipeline_name: str = "custom_pipeline",
    context: dict = None,
    incremental_loading=False,
):
    # Will only be used if ENABLE_BACKEND_ACCESS_CONTROL is set to True
    await set_database_global_context_variables(dataset.id, dataset.owner_id)

    if not data:
        data: list[Data] = await get_dataset_data(dataset_id=dataset.id)

    process_pipeline_status = await process_pipeline_check(dataset, data, pipeline_name)
    if process_pipeline_status:
        # If pipeline was already processed or is currently being processed
        # return status information to async generator and finish execution
        yield process_pipeline_status
        return

    if not isinstance(tasks, list):
        raise ValueError("Tasks must be a list")

    for task in tasks:
        if not isinstance(task, Task):
            raise ValueError(f"Task {task} is not an instance of Task")

    pipeline_run = run_tasks(
        tasks, dataset.id, data, user, pipeline_name, context, incremental_loading
    )

    async for pipeline_run_info in pipeline_run:
        yield pipeline_run_info
