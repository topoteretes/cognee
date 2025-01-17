import asyncio
import logging
from typing import Union

from pydantic import BaseModel

from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.methods import get_datasets, get_datasets_by_name
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.operations.log_pipeline_status import log_pipeline_status
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.utils import send_telemetry
from cognee.tasks.documents import (
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.storage.index_graph_edges import index_graph_edges
from cognee.tasks.summarization import summarize_text

logger = logging.getLogger("cognify.v2")

update_status_lock = asyncio.Lock()


async def cognify(
    datasets: Union[str, list[str]] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    tasks: list[Task] = None,
):
    if user is None:
        user = await get_default_user()

    existing_datasets = await get_datasets(user.id)

    if datasets is None or len(datasets) == 0:
        # If no datasets are provided, cognify all existing datasets.
        datasets = existing_datasets

    if isinstance(datasets[0], str):
        datasets = await get_datasets_by_name(datasets, user.id)

    existing_datasets_map = {
        generate_dataset_name(dataset.name): True for dataset in existing_datasets
    }

    awaitables = []

    if tasks is None:
        tasks = await get_default_tasks(user, graph_model)

    for dataset in datasets:
        dataset_name = generate_dataset_name(dataset.name)

        if dataset_name in existing_datasets_map:
            awaitables.append(run_cognify_pipeline(dataset, user, tasks))

    return await asyncio.gather(*awaitables)


async def run_cognify_pipeline(dataset: Dataset, user: User, tasks: list[Task]):
    data_documents: list[Data] = await get_dataset_data(dataset_id=dataset.id)

    document_ids_str = [str(document.id) for document in data_documents]

    dataset_id = dataset.id
    dataset_name = generate_dataset_name(dataset.name)

    send_telemetry("cognee.cognify EXECUTION STARTED", user.id)

    # async with update_status_lock: TODO: Add UI lock to prevent multiple backend requests
    task_status = await get_pipeline_status([dataset_id])

    if (
        dataset_id in task_status
        and task_status[dataset_id] == PipelineRunStatus.DATASET_PROCESSING_STARTED
    ):
        logger.info("Dataset %s is already being processed.", dataset_name)
        return

    await log_pipeline_status(
        dataset_id,
        PipelineRunStatus.DATASET_PROCESSING_STARTED,
        {
            "dataset_name": dataset_name,
            "files": document_ids_str,
        },
    )

    try:
        if not isinstance(tasks, list):
            raise ValueError("Tasks must be a list")

        for task in tasks:
            if not isinstance(task, Task):
                raise ValueError(f"Task {task} is not an instance of Task")

        pipeline = run_tasks(tasks, data_documents, "cognify_pipeline")

        async for result in pipeline:
            print(result)

        await index_graph_edges()

        send_telemetry("cognee.cognify EXECUTION COMPLETED", user.id)

        await log_pipeline_status(
            dataset_id,
            PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
            {
                "dataset_name": dataset_name,
                "files": document_ids_str,
            },
        )
    except Exception as error:
        send_telemetry("cognee.cognify EXECUTION ERRORED", user.id)

        await log_pipeline_status(
            dataset_id,
            PipelineRunStatus.DATASET_PROCESSING_ERRORED,
            {
                "dataset_name": dataset_name,
                "files": document_ids_str,
            },
        )
        raise error


def generate_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace(".", "_").replace(" ", "_")


async def get_default_tasks(
    user: User = None, graph_model: BaseModel = KnowledgeGraph
) -> list[Task]:
    if user is None:
        user = await get_default_user()

    try:
        cognee_config = get_cognify_config()
        default_tasks = [
            Task(classify_documents),
            Task(check_permissions_on_documents, user=user, permissions=["write"]),
            Task(extract_chunks_from_documents),  # Extract text chunks based on the document type.
            Task(
                extract_graph_from_data, graph_model=graph_model, task_config={"batch_size": 10}
            ),  # Generate knowledge graphs from the document chunks.
            Task(
                summarize_text,
                summarization_model=cognee_config.summarization_model,
                task_config={"batch_size": 10},
            ),
            Task(add_data_points, only_root=True, task_config={"batch_size": 10}),
        ]
    except Exception as error:
        send_telemetry("cognee.cognify DEFAULT TASKS CREATION ERRORED", user.id)
        raise error
    return default_tasks
