import asyncio
import logging
from typing import Union

from cognee.modules.cognify.config import get_cognify_config
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.data.models import Dataset, Data
from cognee.modules.data.operations.get_dataset_data import get_dataset_data
from cognee.modules.data.operations.retrieve_datasets import retrieve_datasets
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.pipelines import run_tasks, run_tasks_parallel
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.operations.log_pipeline_status import log_pipeline_status
from cognee.tasks import chunk_extract_summary, \
    chunk_naive_llm_classifier, \
    chunk_remove_disconnected, \
    infer_data_ontology, \
    save_chunks_to_store, \
    chunk_update_check, \
    chunks_into_graph, \
    source_documents_to_chunks, \
    check_permissions_on_documents, \
    classify_documents

logger = logging.getLogger("cognify.v2")

update_status_lock = asyncio.Lock()

class PermissionDeniedException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

async def cognify(datasets: Union[str, list[str]] = None, user: User = None):
    db_engine = get_relational_engine()

    if datasets is None or len(datasets) == 0:
        return await cognify(await db_engine.get_datasets())

    if type(datasets[0]) == str:
        datasets = await retrieve_datasets(datasets)

    if user is None:
        user = await get_default_user()

    async def run_cognify_pipeline(dataset: Dataset):
        data_documents: list[Data] = await get_dataset_data(dataset_id = dataset.id)

        document_ids_str = [str(document.id) for document in data_documents]

        dataset_id = dataset.id
        dataset_name = generate_dataset_name(dataset.name)

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
                        chunk_extract_summary,
                        summarization_model = cognee_config.summarization_model,
                        collection_name = "chunk_summaries",
                    ), # Summarize the document chunks
                    Task(
                        chunk_naive_llm_classifier,
                        classification_model = cognee_config.classification_model,
                    ),
                ]),
                Task(chunk_remove_disconnected), # Remove the obsolete document chunks.
            ]

            pipeline = run_tasks(tasks, data_documents)

            async for result in pipeline:
                print(result)

            await log_pipeline_status(dataset_id, "DATASET_PROCESSING_FINISHED", {
                "dataset_name": dataset_name,
                "files": document_ids_str,
            })
        except Exception as error:
            await log_pipeline_status(dataset_id, "DATASET_PROCESSING_ERROR", {
                "dataset_name": dataset_name,
                "files": document_ids_str,
            })
            raise error


    existing_datasets = [dataset.name for dataset in list(await db_engine.get_datasets())]
    awaitables = []

    for dataset in datasets:
        dataset_name = generate_dataset_name(dataset.name)

        if dataset_name in existing_datasets:
            awaitables.append(run_cognify_pipeline(dataset))

    return await asyncio.gather(*awaitables)


def generate_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace(".", "_").replace(" ", "_")
