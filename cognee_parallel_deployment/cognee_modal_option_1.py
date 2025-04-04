import modal
import os
import logging
import asyncio
import cognee
import signal
import json
from dotenv import dotenv_values

from cognee.shared.logging_utils import get_logger, ERROR
from cognee.modules.search.types import SearchType
from typing import Optional


from pydantic import BaseModel

from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.pipelines import merge_needs
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.pipelines.tasks.Task import Task, TaskConfig, TaskExecutionCompleted
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.documents import (
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points, index_graph_edges
from cognee.tasks.summarization import summarize_text
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.pipelines.operations.run_tasks import run_tasks_with_telemetry


logger = logging.getLogger("MODAL_DEPLOYED_INSTANCE")

app = modal.App("cognee-runner")

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


async def get_modal_tasks(
    graph_model: BaseModel = KnowledgeGraph,
    ontology_file_path: Optional[str] = None,
) -> list[Task]:
    cognee_config = get_cognify_config()

    ontology_adapter = OntologyResolver(ontology_file=ontology_file_path)

    modal_tasks = [
        Task(  # Generate knowledge graphs from the document chunks.
            extract_graph_from_data,
            graph_model=graph_model,
            ontology_adapter=ontology_adapter,
        ),
        Task(
            summarize_text,
            summarization_model=cognee_config.summarization_model,
        ),
        Task(
            add_data_points,
            index_edges=False,
            task_config=TaskConfig(needs=[merge_needs(summarize_text, extract_graph_from_data)]),
        ),
    ]

    return modal_tasks


@app.function(image=image, max_containers=2, timeout=86400)
async def entry(file, chunk_list):
    print(f"File execution started: {file}")

    modal_tasks = await get_modal_tasks()
    async for _ in run_tasks_with_telemetry(
        modal_tasks, data=chunk_list, pipeline_name=f"modal_execution_file_{file}"
    ):
        pass

    print(f"File execution finished: {file}")


@app.local_entrypoint()
async def main():
    ############MASTER NODE (local for now)
    dataset_name = "dataset_to_parallelize"
    directory_name = "cognee_parallel_deployment/modal_input/"
    batch_size = 200

    # Cleaning the db + adding all the documents to metastore
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(data=os.path.abspath(directory_name), dataset_name=dataset_name)

    # Preparing the dataset
    user = await get_default_user()
    dataset = await get_datasets_by_name(dataset_name, user.id)
    documents = await get_dataset_data(dataset_id=dataset[0].id)

    # Creating chunks
    preprocessing_tasks = await get_preprocessing_steps()
    for i in range(0, len(documents), batch_size):
        data_to_submit = {}
        batch = documents[i : i + batch_size]
        for doc in batch:
            document_name = doc.name
            data_to_submit[document_name] = []

            async for event in run_tasks_with_telemetry(
                preprocessing_tasks, data=[doc], pipeline_name="preprocessing_steps"
            ):
                if (
                    isinstance(event, TaskExecutionCompleted)
                    and event.task is extract_chunks_from_documents
                ):
                    data_to_submit[document_name].extend(event.result)

        print("Batch reading finished...")
        tasks = [entry.remote.aio(file, chunk_list) for file, chunk_list in data_to_submit.items()]

        await asyncio.gather(*tasks, return_exceptions=True)

    await index_graph_edges()
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    logger = get_logger(level=ERROR)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
