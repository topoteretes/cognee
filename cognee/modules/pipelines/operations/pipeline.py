import asyncio
from typing import Union
from uuid import NAMESPACE_OID, uuid5

from cognee.shared.logging_utils import get_logger
from cognee.modules.data.methods import get_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.modules.pipelines.operations import log_pipeline_run_initiated

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

    if not datasets:
        # Get datasets from database if none sent.
        datasets = existing_datasets
    else:
        # If dataset is already in database, use it, otherwise create a new instance.
        dataset_instances = []

        for dataset_name in datasets:
            is_dataset_found = False

            for existing_dataset in existing_datasets:
                if (
                    existing_dataset.name == dataset_name
                    or str(existing_dataset.id) == dataset_name
                ):
                    dataset_instances.append(existing_dataset)
                    is_dataset_found = True
                    break

            if not is_dataset_found:
                dataset_instances.append(
                    Dataset(
                        id=await get_unique_dataset_id(dataset_name=dataset_name, user=user),
                        name=dataset_name,
                        owner_id=user.id,
                    )
                )

        datasets = dataset_instances

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
    check_dataset_name(dataset.name)

    # Ugly hack, but no easier way to do this.
    if pipeline_name == "add_pipeline":
        # Refresh the add pipeline status so data is added to a dataset.
        # Without this the app_pipeline status will be DATASET_PROCESSING_COMPLETED and will skip the execution.
        dataset_id = uuid5(NAMESPACE_OID, f"{dataset.name}{str(user.id)}")

        await log_pipeline_run_initiated(
            pipeline_id=uuid5(NAMESPACE_OID, "add_pipeline"),
            pipeline_name="add_pipeline",
            dataset_id=dataset_id,
        )

        # Refresh the cognify pipeline status after we add new files.
        # Without this the cognify_pipeline status will be DATASET_PROCESSING_COMPLETED and will skip the execution.
        await log_pipeline_run_initiated(
            pipeline_id=uuid5(NAMESPACE_OID, "cognify_pipeline"),
            pipeline_name="cognify_pipeline",
            dataset_id=dataset_id,
        )

    dataset_id = dataset.id

    if not data:
        data: list[Data] = await get_dataset_data(dataset_id=dataset_id)

    # async with update_status_lock: TODO: Add UI lock to prevent multiple backend requests
    if isinstance(dataset, Dataset):
        task_status = await get_pipeline_status([dataset_id], pipeline_name)
    else:
        task_status = [
            PipelineRunStatus.DATASET_PROCESSING_COMPLETED
        ]  # TODO: this is a random assignment, find permanent solution

    if str(dataset_id) in task_status:
        if task_status[str(dataset_id)] == PipelineRunStatus.DATASET_PROCESSING_STARTED:
            logger.info("Dataset %s is already being processed.", dataset_id)
            return
        if task_status[str(dataset_id)] == PipelineRunStatus.DATASET_PROCESSING_COMPLETED:
            logger.info("Dataset %s is already processed.", dataset_id)
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
