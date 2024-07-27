import asyncio
import hashlib
import logging
import uuid
from typing import Union

from fastapi_users import fastapi_users
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.graph import get_graph_config
from cognee.infrastructure.databases.relational.user_authentication.authentication_db import async_session_maker
from cognee.infrastructure.databases.relational.user_authentication.users import has_permission_document, \
    get_user_permissions, get_async_session_context
# from cognee.infrastructure.databases.relational.user_authentication.authentication_db import async_session_maker
# from cognee.infrastructure.databases.relational.user_authentication.users import get_user_permissions, fastapi_users
from cognee.modules.cognify.config import get_cognify_config
from cognee.infrastructure.databases.relational.config import get_relationaldb_config
from cognee.modules.data.processing.document_types.AudioDocument import AudioDocument
from cognee.modules.data.processing.document_types.ImageDocument import ImageDocument
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.data.processing.document_types import PdfDocument, TextDocument
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

class PermissionDeniedException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

async def cognify(datasets: Union[str, list[str]] = None, root_node_id: str = None, user_id:str="default_user"):

    relational_config = get_relationaldb_config()
    db_engine = relational_config.database_engine
    create_task_status_table()

    if datasets is None or len(datasets) == 0:
        return await cognify(db_engine.get_datasets())


    async def run_cognify_pipeline(dataset_name: str, files: list[dict]):

        for file in files:
            file["id"] = str(uuid.uuid4())
            file["name"] = file["name"].replace(" ", "_")

            async with get_async_session_context() as session:

                out = await has_permission_document(user_id, file["id"], "write", session)


                async with update_status_lock:
                    task_status = get_task_status([dataset_name])

                    if dataset_name in task_status and task_status[dataset_name] == "DATASET_PROCESSING_STARTED":
                        logger.info(f"Dataset {dataset_name} is being processed.")
                        return

                    update_task_status(dataset_name, "DATASET_PROCESSING_STARTED")
                try:
                    cognee_config = get_cognify_config()
                    graph_config = get_graph_config()
                    root_node_id = None

                    if graph_config.infer_graph_topology and graph_config.graph_topology_task:
                        from cognee.modules.topology.topology import TopologyEngine
                        topology_engine = TopologyEngine(infer=graph_config.infer_graph_topology)
                        root_node_id = await topology_engine.add_graph_topology(files = files)
                    elif graph_config.infer_graph_topology and not graph_config.infer_graph_topology:
                        from cognee.modules.topology.topology import TopologyEngine
                        topology_engine = TopologyEngine(infer=graph_config.infer_graph_topology)
                        await topology_engine.add_graph_topology(graph_config.topology_file_path)
                    elif not graph_config.graph_topology_task:
                        root_node_id = "ROOT"

                    tasks = [
                        Task(process_documents, parent_node_id = root_node_id, task_config = { "batch_size": 10 }, user_id = hashed_user_id, user_permissions=user_permissions), # Classify documents and save them as a nodes in graph db, extract text chunks based on the document type
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

                    pipeline = run_tasks(tasks, [
                        PdfDocument(title=f"{file['name']}.{file['extension']}", file_path=file["file_path"]) if file["extension"] == "pdf" else
                        AudioDocument(title=f"{file['name']}.{file['extension']}", file_path=file["file_path"]) if file["extension"] == "audio" else
                        ImageDocument(title=f"{file['name']}.{file['extension']}", file_path=file["file_path"]) if file["extension"] == "image" else
                        TextDocument(title=f"{file['name']}.{file['extension']}", file_path=file["file_path"])
                        for file in files
                    ])

                    async for result in pipeline:
                        print(result)

                    update_task_status(dataset_name, "DATASET_PROCESSING_FINISHED")
                except Exception as error:
                    update_task_status(dataset_name, "DATASET_PROCESSING_ERROR")
                    raise error


    existing_datasets = db_engine.get_datasets()

    awaitables = []


    # dataset_files = []
    # dataset_name = datasets.replace(".", "_").replace(" ", "_")

    # for added_dataset in existing_datasets:
    #     if dataset_name in added_dataset:
    #         dataset_files.append((added_dataset, db_engine.get_files_metadata(added_dataset)))

    for dataset in datasets:
        if dataset in existing_datasets:
            # for file_metadata in files:
            #     if root_node_id is None:
            #         root_node_id=file_metadata['id']
            awaitables.append(run_cognify_pipeline(dataset, db_engine.get_files_metadata(dataset)))

    return await asyncio.gather(*awaitables)


#
# if __name__ == "__main__":
#     from cognee.api.v1.add import add
#     from cognee.api.v1.datasets.datasets import datasets
#
#
#     async def aa():
#         await add("TEXT ABOUT NLP AND MONKEYS")
#
#         print(datasets.discover_datasets())
#
#         return



    # asyncio.run(cognify())
