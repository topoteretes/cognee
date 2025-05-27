import asyncio
from typing import Union
from uuid import NAMESPACE_OID, uuid5, UUID

from cognee.exceptions import InvalidValueError
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets
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
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets
from cognee.context_global_variables import set_database_global_context_variables

from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)
from cognee.context_global_variables import (
    graph_db_config as context_graph_db_config,
    vector_db_config as context_vector_db_config,
)

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
):
    # Note: These context variables allow different value assignment for databases in Cognee
    #       per async task, thread, process and etc.
    if vector_db_config:
        context_vector_db_config.set(vector_db_config)
    if graph_db_config:
        context_graph_db_config.set(graph_db_config)

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

    # Get datasets user wants write permissions for (verify user has permissions if datasets are provided as well)
    # NOTE: If a user wants to write to a dataset he does not own it must be provided through UUID
    existing_datasets = await get_existing_datasets(datasets, user)

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
            else:
                raise InvalidValueError(
                    f"Provided dataset is not handled properly: f{dataset_name}"
                )

        datasets = dataset_instances

    awaitables = []

    for dataset in datasets:
        awaitables.append(
            run_pipeline(
                dataset=dataset,
                user=user,
                tasks=tasks,
                data=data,
                pipeline_name=pipeline_name,
                context={"dataset": dataset},
            )
        )

    return await asyncio.gather(*awaitables)


async def run_pipeline(
    dataset: Dataset,
    user: User,
    tasks: list[Task],
    data=None,
    pipeline_name: str = "custom_pipeline",
    context: dict = None,
):
    check_dataset_name(dataset.name)

    # Will only be used if ENABLE_BACKEND_ACCESS_CONTROL is set to True
    await set_database_global_context_variables(dataset.name, user.id)

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

    pipeline_run = run_tasks(tasks, dataset_id, data, user, pipeline_name, context=context)
    pipeline_run_status = None

    async for run_status in pipeline_run:
        pipeline_run_status = run_status

    return pipeline_run_status


def check_dataset_name(dataset_name: str) -> str:
    if "." in dataset_name or " " in dataset_name:
        raise ValueError("Dataset name cannot contain spaces or underscores")


async def get_dataset_ids(datasets: Union[list[str], list[UUID]], user):
    """
    Function returns dataset IDs necessary based on provided input.
    It transforms raw strings into real dataset_ids with keeping write permissions in mind.
    If a user wants to write to a dataset he is not the owner of it must be provided through UUID.
    Args:
        datasets:
        pipeline_name:
        user:

    Returns: a list of write access dataset_ids if they exist

    """
    if all(isinstance(dataset, UUID) for dataset in datasets):
        # Return list of dataset UUIDs
        dataset_ids = datasets
    else:
        # Convert list of dataset names to dataset UUID
        if all(isinstance(dataset, str) for dataset in datasets):
            # Get all user owned dataset objects (If a user wants to write to a dataset he is not the owner of it must be provided through UUID.)
            user_datasets = await get_datasets(user.id)
            # Filter out non name mentioned datasets
            dataset_ids = [dataset.id for dataset in user_datasets if dataset.name in datasets]
        else:
            raise InvalidValueError(f"Provided datasets value is not handled: f{datasets}")

    return dataset_ids


async def get_existing_datasets(
    datasets: Union[list[str], list[UUID]], user: User
) -> list[Dataset]:
    """
    Function returns a list of existing dataset objects user has access for based on datasets input.

    Args:
        datasets:
        user:

    Returns:
        list of Dataset objects

    """
    # TODO: Test 1. add pipeline with: datasetName, datasetName and datasetID
    #       Test 2. Cognify without dataset info, cognify with datasetIDs user has write and no write access for
    if datasets:
        # Function handles transforming dataset input to dataset IDs (if possible)
        dataset_ids = await get_dataset_ids(datasets, user)
        # If dataset_ids are provided filter these datasets based on what user has permission for.
        if dataset_ids:
            existing_datasets = await get_specific_user_permission_datasets(
                user.id, "write", dataset_ids
            )
        else:
            existing_datasets = []
    else:
        # If no datasets are provided, work with all existing datasets user has permission for.
        existing_datasets = await get_all_user_permission_datasets(user, "write")

    return existing_datasets
