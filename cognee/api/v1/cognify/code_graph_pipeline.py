import asyncio
import logging
from typing import Union

from cognee.shared.SourceCodeGraph import SourceCodeGraph
from cognee.shared.utils import send_telemetry
from cognee.modules.data.models import Dataset, Data
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.methods import get_datasets, get_datasets_by_name
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.pipelines import run_tasks
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.operations.log_pipeline_status import log_pipeline_status
from cognee.tasks.documents import classify_documents, check_permissions_on_documents, extract_chunks_from_documents
from cognee.tasks.graph import extract_graph_from_code
from cognee.tasks.storage import add_data_points

logger = logging.getLogger("code_graph_pipeline")

update_status_lock = asyncio.Lock()

async def code_graph_pipeline(datasets: Union[str, list[str]] = None, user: User = None):
    if user is None:
        user = await get_default_user()

    existing_datasets = await get_datasets(user.id)

    if datasets is None or len(datasets) == 0:
        # If no datasets are provided, cognify all existing datasets.
        datasets = existing_datasets

    if type(datasets[0]) == str:
        datasets = await get_datasets_by_name(datasets, user.id)

    existing_datasets_map = {
        generate_dataset_name(dataset.name): True for dataset in existing_datasets
    }

    awaitables = []

    for dataset in datasets:
        dataset_name = generate_dataset_name(dataset.name)

        if dataset_name in existing_datasets_map:
            awaitables.append(run_pipeline(dataset, user))

    return await asyncio.gather(*awaitables)


async def run_pipeline(dataset: Dataset, user: User):
    data_documents: list[Data] = await get_dataset_data(dataset_id = dataset.id)

    document_ids_str = [str(document.id) for document in data_documents]

    dataset_id = dataset.id
    dataset_name = generate_dataset_name(dataset.name)

    send_telemetry("code_graph_pipeline EXECUTION STARTED", user.id)

    async with update_status_lock:
        task_status = await get_pipeline_status([dataset_id])

        if dataset_id in task_status and task_status[dataset_id] == PipelineRunStatus.DATASET_PROCESSING_STARTED:
            logger.info("Dataset %s is already being processed.", dataset_name)
            return

        await log_pipeline_status(dataset_id, PipelineRunStatus.DATASET_PROCESSING_STARTED, {
            "dataset_name": dataset_name,
            "files": document_ids_str,
        })
    try:
        tasks = [
            Task(classify_documents),
            Task(check_permissions_on_documents, user = user, permissions = ["write"]),
            Task(extract_chunks_from_documents), # Extract text chunks based on the document type.
            Task(add_data_points, task_config = { "batch_size": 10 }),
            Task(extract_graph_from_code, graph_model = SourceCodeGraph, task_config = { "batch_size": 10 }), # Generate knowledge graphs from the document chunks.
        ]

        pipeline = run_tasks(tasks, data_documents, "code_graph_pipeline")

        async for result in pipeline:
            print(result)

        send_telemetry("code_graph_pipeline EXECUTION COMPLETED", user.id)

        await log_pipeline_status(dataset_id, PipelineRunStatus.DATASET_PROCESSING_COMPLETED, {
            "dataset_name": dataset_name,
            "files": document_ids_str,
        })
    except Exception as error:
        send_telemetry("code_graph_pipeline EXECUTION ERRORED", user.id)

        await log_pipeline_status(dataset_id, PipelineRunStatus.DATASET_PROCESSING_ERRORED, {
            "dataset_name": dataset_name,
            "files": document_ids_str,
        })
        raise error


def generate_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace(".", "_").replace(" ", "_")
