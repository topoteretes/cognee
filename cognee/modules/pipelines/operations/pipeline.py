import asyncio
from cognee.shared.logging_utils import get_logger
from typing import Union
from uuid import uuid5, NAMESPACE_OID

from cognee.modules.data.methods import get_datasets, get_datasets_by_name
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User

from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)

logger = get_logger("cognee.pipeline")

update_status_lock = asyncio.Lock()


async def cognee_pipeline(
    tasks: list[Task],
    data=None,
    datasets: Union[str, list[str]] = None,
    user: User = None,
    pipeline_name: str = "custom_pipeline",
):
    # Create tables for databases
    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()

    # Initialize first_run attribute if it doesn't exist
    if not hasattr(cognee_pipeline, "first_run"):
        cognee_pipeline.first_run = True

    if cognee_pipeline.first_run:
        from cognee.infrastructure.llm.utils import test_llm_connection, test_embedding_connection

        # Test LLM and Embedding configuration once before running Cognee
        await test_llm_connection()
        await test_embedding_connection()
        cognee_pipeline.first_run = False  # Update flag after first run

    # If no user is provided use default user
    if user is None:
        user = await get_default_user()

    # Convert datasets to list in case it's a string
    if isinstance(datasets, str):
        datasets = [datasets]

    # If no datasets are provided, work with all existing datasets.
    existing_datasets = await get_datasets(user.id)
    if datasets is None or len(datasets) == 0:
        datasets = existing_datasets
        if isinstance(datasets[0], str):
            datasets = await get_datasets_by_name(datasets, user.id)
    else:
        # Try to get datasets objects from database, if they don't exist use dataset name
        datasets_names = await get_datasets_by_name(datasets, user.id)
        if datasets_names:
            datasets = datasets_names

    awaitables = []

    for dataset in datasets:
        awaitables.append(
            run_pipeline(
                dataset=dataset, user=user, tasks=tasks, data=data, pipeline_name=pipeline_name
            )
        )

    return await asyncio.gather(*awaitables)


async def run_pipeline(
    dataset: Dataset,
    user: User,
    tasks: list[Task],
    data=None,
    pipeline_name: str = "custom_pipeline",
):
    if isinstance(dataset, Dataset):
        check_dataset_name(dataset.name)
        dataset_id = dataset.id
    elif isinstance(dataset, str):
        check_dataset_name(dataset)
        # Generate id based on unique dataset_id formula
        dataset_id = uuid5(NAMESPACE_OID, f"{dataset}{str(user.id)}")

    if not data:
        data: list[Data] = await get_dataset_data(dataset_id=dataset_id)

    # async with update_status_lock: TODO: Add UI lock to prevent multiple backend requests
    if isinstance(dataset, Dataset):
        task_status = await get_pipeline_status([dataset_id])
    else:
        task_status = [
            PipelineRunStatus.DATASET_PROCESSING_COMPLETED
        ]  # TODO: this is a random assignment, find permanent solution

    if (
        str(dataset_id) in task_status
        and task_status[str(dataset_id)] == PipelineRunStatus.DATASET_PROCESSING_STARTED
    ):
        logger.info("Dataset %s is already being processed.", dataset_id)
        return

    if not isinstance(tasks, list):
        raise ValueError("Tasks must be a list")

    for task in tasks:
        if not isinstance(task, Task):
            raise ValueError(f"Task {task} is not an instance of Task")

    pipeline_run = run_tasks(tasks, dataset_id, data, user, pipeline_name)
    pipeline_run_status = None

    async for run_status in pipeline_run:
        pipeline_run_status = run_status

    return pipeline_run_status


def check_dataset_name(dataset_name: str) -> str:
    if "." in dataset_name or " " in dataset_name:
        raise ValueError("Dataset name cannot contain spaces or underscores")
