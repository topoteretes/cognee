import asyncio
from uuid import UUID
from typing import Union

from cognee.shared.logging_utils import get_logger
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.utils import generate_pipeline_id

from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.modules.pipelines.operations import log_pipeline_run_initiated
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.pipelines.layers.authorized_user_datasets import authorized_user_datasets
from cognee.modules.pipelines.layers.pipeline_status_check import pipeline_status_check

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
    incremental_loading: bool = False,
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
        from cognee.infrastructure.llm.utils import (
            test_llm_connection,
            test_embedding_connection,
        )

        # Test LLM and Embedding configuration once before running Cognee
        await test_llm_connection()
        await test_embedding_connection()
        cognee_pipeline.first_run = False  # Update flag after first run

    user, authorized_datasets = await authorized_user_datasets(user, datasets)

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

    if not data:
        data: list[Data] = await get_dataset_data(dataset_id=dataset.id)

    async for pipeline_status in pipeline_status_check(dataset, data, pipeline_name):
        yield pipeline_status

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
