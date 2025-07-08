import asyncio
from pydantic import BaseModel
from typing import Union, Optional

from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm import get_max_chunk_tokens

from cognee.modules.pipelines import cognee_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunErrored
from cognee.modules.pipelines.queues.pipeline_run_info_queues import push_to_queue
from cognee.modules.users.models import User

from cognee.tasks.documents import (
    check_permissions_on_dataset,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text

logger = get_logger("cognify")

update_status_lock = asyncio.Lock()


async def cognify(
    datasets: Union[str, list[str]] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    run_in_background: bool = False,
):
    tasks = await get_default_tasks(user, graph_model, chunker, chunk_size, ontology_file_path)

    if run_in_background:
        return await run_cognify_as_background_process(
            tasks=tasks,
            user=user,
            datasets=datasets,
            vector_db_config=vector_db_config,
            graph_db_config=graph_db_config,
        )
    else:
        return await run_cognify_blocking(
            tasks=tasks,
            user=user,
            datasets=datasets,
            vector_db_config=vector_db_config,
            graph_db_config=graph_db_config,
        )


async def run_cognify_blocking(
    tasks,
    user,
    datasets,
    graph_db_config: dict = None,
    vector_db_config: dict = False,
):
    total_run_info = {}

    async for run_info in cognee_pipeline(
        tasks=tasks,
        datasets=datasets,
        user=user,
        pipeline_name="cognify_pipeline",
        graph_db_config=graph_db_config,
        vector_db_config=vector_db_config,
    ):
        if run_info.dataset_id:
            total_run_info[run_info.dataset_id] = run_info
        else:
            total_run_info = run_info

    return total_run_info


async def run_cognify_as_background_process(
    tasks,
    user,
    datasets,
    graph_db_config: dict = None,
    vector_db_config: dict = False,
):
    # Store pipeline status for all pipelines
    pipeline_run_started_info = []

    async def handle_rest_of_the_run(pipeline_list):
        # Execute all provided pipelines one by one to avoid database write conflicts
        for pipeline in pipeline_list:
            while True:
                try:
                    pipeline_run_info = await anext(pipeline)

                    push_to_queue(pipeline_run_info.pipeline_run_id, pipeline_run_info)

                    if isinstance(pipeline_run_info, PipelineRunCompleted) or isinstance(
                        pipeline_run_info, PipelineRunErrored
                    ):
                        break
                except StopAsyncIteration:
                    break

    # Start all pipelines to get started status
    pipeline_list = []
    for dataset in datasets:
        pipeline_run = cognee_pipeline(
            tasks=tasks,
            user=user,
            datasets=dataset,
            pipeline_name="cognify_pipeline",
            graph_db_config=graph_db_config,
            vector_db_config=vector_db_config,
        )

        # Save dataset Pipeline run started info
        pipeline_run_started_info.append(await anext(pipeline_run))
        pipeline_list.append(pipeline_run)

    # Send all started pipelines to execute one by one in background
    asyncio.create_task(handle_rest_of_the_run(pipeline_list=pipeline_list))

    return pipeline_run_started_info


async def get_default_tasks(  # TODO: Find out a better way to do this (Boris's comment)
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
) -> list[Task]:
    default_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_dataset, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),  # Extract text chunks based on the document type.
        Task(
            extract_graph_from_data,
            graph_model=graph_model,
            ontology_adapter=OntologyResolver(ontology_file=ontology_file_path),
            task_config={"batch_size": 10},
        ),  # Generate knowledge graphs from the document chunks.
        Task(
            summarize_text,
            task_config={"batch_size": 10},
        ),
        Task(add_data_points, task_config={"batch_size": 10}),
    ]

    return default_tasks
