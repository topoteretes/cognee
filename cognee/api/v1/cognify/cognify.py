import asyncio
from cognee.shared.logging_utils import get_logger
from typing import Union, Optional

from pydantic import BaseModel

from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.methods import get_datasets, get_datasets_by_name
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.documents import (
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.modules.chunking.TextChunker import TextChunker

logger = get_logger("cognify")

update_status_lock = asyncio.Lock()


async def cognify(
    datasets: Union[str, list[str]] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    tasks: list[Task] = None,
    ontology_file_path: Optional[str] = None,
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
        tasks = await get_default_tasks(user, graph_model, ontology_file_path=ontology_file_path)

    for dataset in datasets:
        dataset_name = generate_dataset_name(dataset.name)

        if dataset_name in existing_datasets_map:
            awaitables.append(run_cognify_pipeline(dataset, user, tasks))

    return await asyncio.gather(*awaitables)


async def run_cognify_pipeline(dataset: Dataset, user: User, tasks: list[Task]):
    data_documents: list[Data] = await get_dataset_data(dataset_id=dataset.id)

    dataset_id = dataset.id
    dataset_name = generate_dataset_name(dataset.name)

    # async with update_status_lock: TODO: Add UI lock to prevent multiple backend requests
    task_status = await get_pipeline_status([dataset_id])

    if (
        str(dataset_id) in task_status
        and task_status[str(dataset_id)] == PipelineRunStatus.DATASET_PROCESSING_STARTED
    ):
        logger.info("Dataset %s is already being processed.", dataset_name)
        return

    if not isinstance(tasks, list):
        raise ValueError("Tasks must be a list")

    for task in tasks:
        if not isinstance(task, Task):
            raise ValueError(f"Task {task} is not an instance of Task")

    pipeline_run = run_tasks(tasks, dataset.id, data_documents, "cognify_pipeline")
    pipeline_run_status = None

    async for run_status in pipeline_run:
        pipeline_run_status = run_status

    return pipeline_run_status


def generate_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace(".", "_").replace(" ", "_")


async def get_default_tasks(  # TODO: Find out a better way to do this (Boris's comment)
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
) -> list[Task]:
    if user is None:
        user = await get_default_user()

    cognee_config = get_cognify_config()

    ontology_adapter = OntologyResolver(ontology_file=ontology_file_path)

    default_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_documents, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),  # Extract text chunks based on the document type.
        Task(
            extract_graph_from_data,
            graph_model=graph_model,
            ontology_adapter=ontology_adapter,
            task_config={"batch_size": 10},
        ),  # Generate knowledge graphs from the document chunks.
        Task(
            summarize_text,
            summarization_model=cognee_config.summarization_model,
            task_config={"batch_size": 10},
        ),
        Task(add_data_points, task_config={"batch_size": 10}),
    ]

    return default_tasks
