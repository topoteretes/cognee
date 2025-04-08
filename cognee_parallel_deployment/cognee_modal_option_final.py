import time
import random
import modal
from modal import App, Queue
import cognee
import asyncio
import os
from dotenv import dotenv_values
import logging
import json
from typing import Optional
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph import get_graph_engine


from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger, ERROR
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.pipelines.tasks.Task import Task, TaskConfig, TaskExecutionCompleted
from cognee.tasks.documents import (
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
)
from pydantic import BaseModel
from cognee.tasks.graph import extract_graph_from_data
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines import merge_needs
from cognee.tasks.storage import add_data_points, index_graph_edges
from cognee.tasks.summarization import summarize_text
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.pipelines.operations.run_tasks import run_tasks_with_telemetry
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.graph.utils import deduplicate_nodes_and_edges, get_graph_from_model
from cognee.tasks.storage.index_data_points import index_data_points


# ------------------------------------------------------------------------------
# App and Queue Initialization
# ------------------------------------------------------------------------------

# Initialize the Modal application
app = App("Cognee_modal_parallel_final")
logger = logging.getLogger("MODAL_DEPLOYED_INSTANCE")

local_env_vars = dict(dotenv_values(".env"))
logger.info("Modal deployment started with the following environmental variables:")
logger.info(json.dumps(local_env_vars, indent=4))

image = (
    modal.Image.from_dockerfile(path="Dockerfile_modal", force_build=False)
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .add_local_file("poetry.lock", remote_path="/root/poetry.lock", copy=True)
    .env(local_env_vars)
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
    .pip_install("protobuf", "h2", "neo4j", "asyncpg", "pgvector")
    .add_local_python_source("cognee")
)


# Create (or get) two queues:
# - graph_nodes_and_edges: Stores messages produced by the producer functions.
# - finished_producers: Keeps track of the number of finished producer jobs.
graph_nodes_and_edges = Queue.from_name("graph_nodes_and_edges", create_if_missing=True)

finished_producers = Queue.from_name("finished_producers", create_if_missing=True)

# ------------------------------------------------------------------------------
# Cognee pipeline steps
# ------------------------------------------------------------------------------


# Preprocessing steps. This gets called in the entrypoint
async def get_preprocessing_steps(user: User = None, chunker=TextChunker) -> list[Task]:
    if user is None:
        user = await get_default_user()

    preprocessing_tasks = [
        Task(classify_documents),
        Task(
            check_permissions_on_documents,
            user=user,
            permissions=["write"],
            task_config=TaskConfig(needs=[classify_documents]),
        ),
        Task(  # Extract text chunks based on the document type.
            extract_chunks_from_documents,
            max_chunk_size=None or get_max_chunk_tokens(),
            chunker=chunker,
            task_config=TaskConfig(needs=[check_permissions_on_documents], output_batch_size=10),
        ),
    ]

    return preprocessing_tasks


# This is the last step of the pipeline that gets executed on modal executors (functions)
async def queue_merge(data_points: list = None, data_point_connections: list = None):
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

    print(f"nodes: {nodes}")

    print(f"edges: {edges}")

    print(f"data_point_connections: {data_point_connections}")

    graph_nodes_and_edges.put((nodes, edges, data_point_connections))

    return nodes, edges, data_point_connections


# This is the pipeline for the modal executors
async def get_graph_tasks_parallelized(
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
        Task(
            queue_merge,
            task_config=TaskConfig(needs=[merge_needs(summarize_text, extract_graph_from_data)]),
        ),
    ]

    return step_two_tasks


# ------------------------------------------------------------------------------
# Producer Function
# ------------------------------------------------------------------------------


@app.function(image=image, timeout=86400, max_containers=100)
async def producer(file_name: str, chunk_list: list):
    print(f"File execution started: {file_name}")
    print(chunk_list)
    modal_tasks = await get_graph_tasks_parallelized()
    async for _ in run_tasks_with_telemetry(
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
        print("Beginning_of_while_true...")
        if graph_nodes_and_edges.len() != 0:
            # COGNEE STEPS GRAPH INGESTION
            print(f"queue len before ingestion{graph_nodes_and_edges.len()}")
            nodes_and_edges = graph_nodes_and_edges.get(block=False)
            if nodes_and_edges is not None:
                if nodes_and_edges[0] is not None:
                    await graph_engine.add_nodes(nodes_and_edges[0])
                if nodes_and_edges[1] is not None:
                    await graph_engine.add_edges(nodes_and_edges[1])
                if nodes_and_edges[2] is not None:
                    await graph_engine.add_edges(nodes_and_edges[2])
            else:
                print(f"nodes_and_edges is {nodes_and_edges}")

            print(f"queue len after ingestion{graph_nodes_and_edges.len()}")
        else:
            await asyncio.sleep(5)
            print("Polling the queue but its empty")
            number_of_finished_jobs = finished_producers.get(block=False)
            if number_of_finished_jobs == number_of_files:
                # We put it back for the other consumers to see that we finished
                finished_producers.put(number_of_finished_jobs)
                print("Finished processing all input elements; stopping consumers.")
                return True


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------


@app.local_entrypoint()
async def main():
    graph_nodes_and_edges.clear()
    finished_producers.clear()
    dataset_name = "dataset_to_parallelize"
    directory_name = "cognee_parallel_deployment/modal_input/"
    number_of_consumers = 1  # Total number of consumer functions to spawn
    batch_size = 300  # Batch size for producers
    results = []
    consumer_futures = []

    # Delete DBs and add the files to the metastore
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(data=os.path.abspath(directory_name), dataset_name=dataset_name)

    user = await get_default_user()
    dataset = await get_datasets_by_name(dataset_name, user.id)
    documents = await get_dataset_data(dataset_id=dataset[0].id)

    print(f"We have {documents} documents in the dataset.")

    # Defining preprocessing tasks ()
    preprocessing_tasks = await get_preprocessing_steps()
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
            async for event in run_tasks_with_telemetry(
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
