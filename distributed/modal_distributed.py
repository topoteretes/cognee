import os
import json
import pathlib
import asyncio
from typing import Optional
from pydantic import BaseModel
from dotenv import dotenv_values
from cognee.modules.chunking.models import DocumentChunk
from modal import App, Queue, Image

import cognee
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.pipelines.operations import run_tasks
from cognee.modules.users.methods import get_default_user
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.cognify.config import get_cognify_config
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.graph.utils import deduplicate_nodes_and_edges, get_graph_from_model
from cognee.modules.pipelines.tasks import Task


# Global tasks
from cognee.tasks.documents import (
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.storage.index_data_points import index_data_points

# Local tasks
from .tasks.extract_graph_from_data import extract_graph_from_data
from .tasks.summarize_text import summarize_text


# ------------------------------------------------------------------------------
# App and Queue Initialization
# ------------------------------------------------------------------------------

# Initialize the Modal application
app = App("cognee_modal_distributed")
logger = get_logger("cognee_modal_distributed")

local_env_vars = dict(dotenv_values(".env"))
logger.info("Modal deployment started with the following environmental variables:")
logger.info(json.dumps(local_env_vars, indent=4))

image = (
    Image.from_dockerfile(path="Dockerfile_modal", force_build=False)
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .add_local_file("poetry.lock", remote_path="/root/poetry.lock", copy=True)
    .env(local_env_vars)
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
    # .pip_install("protobuf", "h2", "neo4j", "asyncpg", "pgvector")
    .add_local_python_source("../cognee")
)


# Create (or get) two queues:
# - graph_nodes_and_edges: Stores messages produced by the producer functions.
# - finished_producers: Keeps track of the number of finished producer jobs.
graph_nodes_and_edges = Queue.from_name("graph_nodes_and_edges", create_if_missing=True)

finished_producers = Queue.from_name("finished_producers", create_if_missing=True)

# ------------------------------------------------------------------------------
# Cognee pipeline steps
# ------------------------------------------------------------------------------


def add_data_to_save_queue(document_chunks: list[DocumentChunk]):
    future = producer.spawn(file_name=document_name, chunk_list=event.result)
    futures.append(future)

# Preprocessing steps. This gets called in the entrypoint
async def get_preprocessing_steps(chunker=TextChunker) -> list[Task]:
    preprocessing_tasks = [
        Task(classify_documents),
        Task(  # Extract text chunks based on the document type.
            extract_chunks_from_documents,
            max_chunk_size=None or get_max_chunk_tokens(),
            chunker=chunker,
        ),
        Task(
            add_data_to_save_queue,
            task_config={"batch_size": 50},
        ),
    ]

    return preprocessing_tasks


# This is the last step of the pipeline that gets executed on modal executors (functions)
async def save_data_points(data_points: list = None, data_point_connections: list = None):
    data_point_connections = data_point_connections or []

    nodes = []
    edges = []

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    results = await asyncio.gather(
        *[
            get_graph_from_model(
                data_point,
                added_nodes=added_nodes,
                added_edges=added_edges,
                visited_properties=visited_properties,
            )
            for data_point in data_points
        ]
    )

    for result_nodes, result_edges in results:
        nodes.extend(result_nodes)
        edges.extend(result_edges)

    nodes, edges = deduplicate_nodes_and_edges(nodes, edges)

    await index_data_points(nodes)

    graph_nodes_and_edges.put((nodes, edges + data_point_connections))


# This is the pipeline for the modal executors
async def get_graph_tasks(
    graph_model: BaseModel = KnowledgeGraph,
    ontology_file_path: Optional[str] = None,
) -> list[Task]:
    cognee_config = get_cognify_config()

    ontology_adapter = OntologyResolver(ontology_file=ontology_file_path)

    step_two_tasks = [
        Task(
            extract_graph_from_data,
            graph_model=graph_model,
            ontology_adapter=ontology_adapter,
        ),
        Task(
            summarize_text,
            summarization_model=cognee_config.summarization_model,
        ),
        Task(save_data_points),
    ]

    return step_two_tasks


# ------------------------------------------------------------------------------
# Producer Function
# ------------------------------------------------------------------------------


@app.function(image=image, timeout=86400, max_containers=100)
async def producer(file_name: str, chunk_list: list):
    modal_tasks = await get_graph_tasks()
    async for _ in run_tasks(
        modal_tasks, data=chunk_list, pipeline_name=f"modal_execution_file_{file_name}"
    ):
        pass

    print(f"File execution finished: {file_name}")

    return file_name


# ------------------------------------------------------------------------------
# Consumer Function
# ------------------------------------------------------------------------------


@app.function(image=image, timeout=86400, max_containers=100)
async def consumer(number_of_files: int):
    graph_engine = await get_graph_engine()

    while True:
        if graph_nodes_and_edges.len() != 0:
            nodes_and_edges = graph_nodes_and_edges.get(block=False)
            if nodes_and_edges is not None:
                if nodes_and_edges[0] is not None:
                    await graph_engine.add_nodes(nodes_and_edges[0])
                if nodes_and_edges[1] is not None:
                    await graph_engine.add_edges(nodes_and_edges[1])
            else:
                print(f"Nodes and edges are: {nodes_and_edges}")
        else:
            await asyncio.sleep(5)

            number_of_finished_jobs = finished_producers.get(block=False)

            if number_of_finished_jobs == number_of_files:
                # We put it back for the other consumers to see that we finished
                finished_producers.put(number_of_finished_jobs)

                print("Finished processing all nodes and edges; stopping graph engine queue.")
                return True


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------


@app.local_entrypoint()
async def main():
    # Clear queues
    graph_nodes_and_edges.clear()
    finished_producers.clear()

    dataset_name = "main"
    data_directory_name = ".data"
    data_directory_path = os.path.join(pathlib.Path(__file__).parent, data_directory_name)

    number_of_consumers = 1  # Total number of consumer functions to spawn
    batch_size = 50  # Batch size for producers

    results = []
    consumer_futures = []

    # Delete DBs and saved files from metastore
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Add files to the metastore
    await cognee.add(data=data_directory_path, dataset_name=dataset_name)

    user = await get_default_user()
    datasets = await get_datasets_by_name(dataset_name, user.id)
    documents = await get_dataset_data(dataset_id=datasets[0].id)

    print(f"We have {len(documents)} documents in the dataset.")

    preprocessing_tasks = await get_preprocessing_steps(user)

    # Start consumer functions
    for _ in range(number_of_consumers):
        consumer_future = consumer.spawn(number_of_files=len(documents))
        consumer_futures.append(consumer_future)

    # Process producer jobs in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        futures = []
        for item in batch:
            document_name = item.name
            async for event in run_tasks(
                preprocessing_tasks, data=[item], pipeline_name="preprocessing_steps"
            ):
                if (
                    isinstance(event, TaskExecutionCompleted)
                    and event.task is extract_chunks_from_documents
                ):
                    future = producer.spawn(file_name=document_name, chunk_list=event.result)
                    futures.append(future)

        batch_results = []
        for future in futures:
            try:
                result = future.get()
            except Exception as e:
                result = e
            batch_results.append(result)

        results.extend(batch_results)
        finished_producers.put(len(results))

    for consumer_future in consumer_futures:
        try:
            print("Finished but waiting")
            consumer_final = consumer_future.get()
            print(f"We got all futures{consumer_final}")
        except Exception as e:
            print(e)
            pass

    print(results)
