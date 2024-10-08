import asyncio
import logging
from typing import Union

from cognee.shared.utils import send_telemetry
from cognee.modules.cognify.config import get_cognify_config
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.data.models import Dataset, Data
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.methods import get_datasets, get_datasets_by_name
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.pipelines import run_tasks, run_tasks_parallel
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.operations.log_pipeline_status import log_pipeline_status
from cognee.tasks import chunk_naive_llm_classifier, \
    chunk_remove_disconnected, \
    infer_data_ontology, \
    save_chunks_to_store, \
    chunk_update_check, \
    chunks_into_graph, \
    source_documents_to_chunks, \
    check_permissions_on_documents, \
    classify_documents
from cognee.tasks.summarization import summarize_text

logger = logging.getLogger("cognify.v2")

update_status_lock = asyncio.Lock()

class PermissionDeniedException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

async def cognify(datasets: Union[str, list[str]] = None, user: User = None):
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
            awaitables.append(run_cognify_pipeline(dataset, user))

    return await asyncio.gather(*awaitables)


async def run_cognify_pipeline(dataset: Dataset, user: User):
    data_documents: list[Data] = await get_dataset_data(dataset_id = dataset.id)

    document_ids_str = [str(document.id) for document in data_documents]

    dataset_id = dataset.id
    dataset_name = generate_dataset_name(dataset.name)

    send_telemetry("cognee.cognify EXECUTION STARTED", user.id)

    async with update_status_lock:
        task_status = await get_pipeline_status([dataset_id])

        if dataset_id in task_status and task_status[dataset_id] == "DATASET_PROCESSING_STARTED":
            logger.info("Dataset %s is already being processed.", dataset_name)
            return

        await log_pipeline_status(dataset_id, "DATASET_PROCESSING_STARTED", {
            "dataset_name": dataset_name,
            "files": document_ids_str,
        })
    try:
        cognee_config = get_cognify_config()

        root_node_id = None

        tasks = [
            Task(classify_documents),
            Task(check_permissions_on_documents, user = user, permissions = ["write"]),
            Task(infer_data_ontology, root_node_id = root_node_id, ontology_model = KnowledgeGraph),
            Task(source_documents_to_chunks, parent_node_id = root_node_id), # Classify documents and save them as a nodes in graph db, extract text chunks based on the document type
            Task(chunks_into_graph, graph_model = KnowledgeGraph, collection_name = "entities", task_config = { "batch_size": 10 }), # Generate knowledge graphs from the document chunks and attach it to chunk nodes
            Task(chunk_update_check, collection_name = "chunks"), # Find all affected chunks, so we don't process unchanged chunks
            Task(
                save_chunks_to_store,
                collection_name = "chunks",
            ), # Save the document chunks in vector db and as nodes in graph db (connected to the document node and between each other)
            run_tasks_parallel([
                Task(
                    summarize_text,
                    summarization_model = cognee_config.summarization_model,
                    collection_name = "summaries",
                ),
                Task(
                    chunk_naive_llm_classifier,
                    classification_model = cognee_config.classification_model,
                ),
            ]),
            Task(chunk_remove_disconnected), # Remove the obsolete document chunks.
        ]

        pipeline = run_tasks(tasks, data_documents, "cognify_pipeline")

        async for result in pipeline:
            print(result)

        send_telemetry("cognee.cognify EXECUTION COMPLETED", user.id)

        await log_pipeline_status(dataset_id, "DATASET_PROCESSING_COMPLETED", {
            "dataset_name": dataset_name,
            "files": document_ids_str,
        })
    except Exception as error:
        send_telemetry("cognee.cognify EXECUTION ERRORED", user.id)

        await log_pipeline_status(dataset_id, "DATASET_PROCESSING_ERRORED", {
            "dataset_name": dataset_name,
            "files": document_ids_str,
        })
        raise error


def generate_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace(".", "_").replace(" ", "_")
