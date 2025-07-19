import asyncio
from uuid import UUID
from typing import Union

from cognee.shared.logging_utils import get_logger
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.methods import get_pipeline_run_by_dataset

from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.modules.pipelines.operations import log_pipeline_run_initiated
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import (
    get_authorized_existing_datasets,
    load_or_create_datasets,
    check_dataset_name,
)

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunStarted,
)

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

    # Convert datasets to list
    if isinstance(datasets, str) or isinstance(datasets, UUID):
        datasets = [datasets]

    # Get datasets user wants write permissions for (verify user has permissions if datasets are provided as well)
    # NOTE: If a user wants to write to a dataset he does not own it must be provided through UUID
    existing_datasets = await get_authorized_existing_datasets(datasets, "write", user)

    if not datasets:
        # Get datasets from database if none sent.
        datasets = existing_datasets
    else:
        # If dataset matches an existing Dataset (by name or id), reuse it. Otherwise, create a new Dataset.
        datasets = await load_or_create_datasets(datasets, existing_datasets, user)

    if not datasets:
        raise DatasetNotFoundError("There are no datasets to work with.")

    for dataset in datasets:
        async for run_info in run_pipeline(
            dataset=dataset,
            user=user,
            tasks=tasks,
            data=data,
            pipeline_name=pipeline_name,
            context={"dataset": dataset},
        ):
            yield run_info


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
    await set_database_global_context_variables(dataset.id, dataset.owner_id)

    # Ugly hack, but no easier way to do this.
    if pipeline_name == "add_pipeline":
        pipeline_id = generate_pipeline_id(user.id, dataset.id, pipeline_name)
        # Refresh the add pipeline status so data is added to a dataset.
        # Without this the app_pipeline status will be DATASET_PROCESSING_COMPLETED and will skip the execution.

        await log_pipeline_run_initiated(
            pipeline_id=pipeline_id,
            pipeline_name="add_pipeline",
            dataset_id=dataset.id,
        )

        # Refresh the cognify pipeline status after we add new files.
        # Without this the cognify_pipeline status will be DATASET_PROCESSING_COMPLETED and will skip the execution.
        await log_pipeline_run_initiated(
            pipeline_id=pipeline_id,
            pipeline_name="cognify_pipeline",
            dataset_id=dataset.id,
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
            pipeline_run = await get_pipeline_run_by_dataset(dataset_id, pipeline_name)
            yield PipelineRunStarted(
                pipeline_run_id=pipeline_run.pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                payload=data,
            )
            return
        elif task_status[str(dataset_id)] == PipelineRunStatus.DATASET_PROCESSING_COMPLETED:
            logger.info("Dataset %s is already processed.", dataset_id)
            pipeline_run = await get_pipeline_run_by_dataset(dataset_id, pipeline_name)
            yield PipelineRunCompleted(
                pipeline_run_id=pipeline_run.pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
            )
            return

    if not isinstance(tasks, list):
        raise ValueError("Tasks must be a list")

    for task in tasks:
        if not isinstance(task, Task):
            raise ValueError(f"Task {task} is not an instance of Task")

    pipeline_run = run_tasks(tasks, dataset_id, data, user, pipeline_name, context)

    async for pipeline_run_info in pipeline_run:
        yield pipeline_run_info
