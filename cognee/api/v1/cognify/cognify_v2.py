import asyncio
import logging
from typing import Union
from cognee.modules.cognify.config import get_cognify_config
from cognee.infrastructure.databases.relational.config import get_relationaldb_config
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.data.processing.document_types.PdfDocument import PdfDocument
from cognee.modules.cognify.vector import save_data_chunks
from cognee.modules.data.processing.process_documents import process_documents
from cognee.modules.classification.classify_text_chunks import classify_text_chunks
from cognee.modules.data.extraction.data_summary.summarize_text_chunks import summarize_text_chunks
from cognee.modules.data.processing.filter_affected_chunks import filter_affected_chunks
from cognee.modules.data.processing.remove_obsolete_chunks import remove_obsolete_chunks
from cognee.modules.data.extraction.knowledge_graph.expand_knowledge_graph import expand_knowledge_graph
from cognee.modules.data.extraction.knowledge_graph.establish_graph_topology import establish_graph_topology
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.pipelines import run_tasks, run_tasks_parallel
from cognee.modules.tasks import create_task_status_table, update_task_status, get_task_status

logger = logging.getLogger("cognify.v2")

update_status_lock = asyncio.Lock()

async def cognify(datasets: Union[str, list[str]] = None):
    relational_config = get_relationaldb_config()
    db_engine = relational_config.database_engine
    create_task_status_table()

    if datasets is None or len(datasets) == 0:
        return await cognify(db_engine.get_datasets())


    async def run_cognify_pipeline(dataset_name: str, files: list[dict]):
        async with update_status_lock:
            task_status = get_task_status([dataset_name])

            if dataset_name in task_status and task_status[dataset_name] == "DATASET_PROCESSING_STARTED":
                logger.info(f"Dataset {dataset_name} is being processed.")
                return

            update_task_status(dataset_name, "DATASET_PROCESSING_STARTED")
        try:
            cognee_config = get_cognify_config()

            tasks = [
                Task(process_documents, parent_node_id = "Boris's documents", task_config = { "batch_size": 10 }), # Classify documents and save them as a nodes in graph db, extract text chunks based on the document type
                Task(establish_graph_topology, topology_model = KnowledgeGraph), # Set the graph topology for the document chunk data
                Task(expand_knowledge_graph, graph_model = KnowledgeGraph), # Generate knowledge graphs from the document chunks and attach it to chunk nodes
                Task(filter_affected_chunks, collection_name = "chunks"), # Find all affected chunks, so we don't process unchanged chunks
                Task(
                    save_data_chunks,
                    collection_name = "chunks",
                ), # Save the document chunks in vector db and as nodes in graph db (connected to the document node and between each other)
                run_tasks_parallel([
                    Task(
                        summarize_text_chunks,
                        summarization_model = cognee_config.summarization_model,
                        collection_name = "chunk_summaries",
                    ), # Summarize the document chunks
                    Task(
                        classify_text_chunks,
                        classification_model = cognee_config.classification_model,
                    ),
                ]),
                Task(remove_obsolete_chunks), # Remove the obsolete document chunks.
            ]

            pipeline = run_tasks(tasks, [PdfDocument(title = file["name"], file_path = file["file_path"]) for file in files])

            async for result in pipeline:
                print(result)

            update_task_status(dataset_name, "DATASET_PROCESSING_FINISHED")
        except Exception as error:
            update_task_status(dataset_name, "DATASET_PROCESSING_ERROR")
            raise error


    existing_datasets = db_engine.get_datasets()

    awaitables = []

    for dataset in datasets:
        if dataset in existing_datasets:
            awaitables.append(run_cognify_pipeline(dataset, db_engine.get_files_metadata(dataset)))

    return await asyncio.gather(*awaitables)
