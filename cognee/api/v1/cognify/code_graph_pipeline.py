# NOTICE: This module contains deprecated functions.
# Use only the run_code_graph_pipeline function; all other functions are deprecated.
# Related issue: COG-906

import asyncio
import logging
from pathlib import Path
from typing import Union

from cognee.modules.data.methods import get_datasets, get_datasets_by_name
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import \
    get_pipeline_status
from cognee.modules.pipelines.operations.log_pipeline_status import \
    log_pipeline_status
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.SourceCodeGraph import SourceCodeGraph
from cognee.shared.utils import send_telemetry
from cognee.tasks.documents import (check_permissions_on_documents,
                                    classify_documents,
                                    extract_chunks_from_documents)
from cognee.tasks.graph import extract_graph_from_code
from cognee.tasks.repo_processor import (enrich_dependency_graph,
                                         expand_dependency_graph,
                                         get_repo_file_dependencies)
from cognee.tasks.storage import add_data_points

from cognee.base_config import get_base_config
from cognee.shared.data_models import MonitoringTool
if MonitoringTool.LANGFUSE:
    from langfuse.decorators import observe

from cognee.tasks.summarization import summarize_code


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

@observe
async def run_pipeline(dataset: Dataset, user: User):
    '''DEPRECATED: Use `run_code_graph_pipeline` instead. This function will be removed.'''
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


async def run_code_graph_pipeline(repo_path):
    import os
    import pathlib
    import cognee
    from cognee.infrastructure.databases.relational import create_db_and_tables

    file_path = Path(__file__).parent
    data_directory_path = str(pathlib.Path(os.path.join(file_path, ".data_storage/code_graph")).resolve())
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(pathlib.Path(os.path.join(file_path, ".cognee_system/code_graph")).resolve())
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await create_db_and_tables()

    tasks = [
        Task(get_repo_file_dependencies),
        Task(enrich_dependency_graph, task_config={"batch_size": 50}),
        Task(expand_dependency_graph, task_config={"batch_size": 50}),
        Task(summarize_code, task_config={"batch_size": 50}),
        Task(add_data_points, task_config={"batch_size": 50}),
    ]

    return run_tasks(tasks, repo_path, "cognify_code_pipeline")
