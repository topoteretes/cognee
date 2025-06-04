import asyncio
from pydantic import BaseModel
from typing import Union, Optional, Any

from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.users.models import User
from cognee.modules.pipelines import cognee_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunStarted
from cognee.modules.pipelines.queues.pipeline_run_info_queues import push_to_queue
from cognee.modules.graph.operations import get_formatted_graph_data
from cognee.modules.crewai.get_crewai_pipeline_run_id import get_crewai_pipeline_run_id

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
    datapoints: dict[str, Any] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    run_in_background: bool = False,
    is_stream_info_enabled: bool = False,
    pipeline_name: str = "cognify_pipeline",
):
    tasks = await get_default_tasks(
        user=user,
        graph_model=graph_model,
        chunker=chunker,
        chunk_size=chunk_size,
        ontology_file_path=ontology_file_path,
    )

    if not user:
        user = await get_default_user()

    if run_in_background:
        return await run_cognify_as_background_process(
            tasks=tasks,
            user=user,
            datasets=datasets,
            datapoints=datapoints,
            pipeline_name=pipeline_name,
            vector_db_config=vector_db_config,
            graph_db_config=graph_db_config,
        )
    else:
        return await run_cognify_blocking(
            tasks=tasks,
            user=user,
            datasets=datasets,
            pipeline_name=pipeline_name,
            datapoints=datapoints,
            is_stream_info_enabled=is_stream_info_enabled,
            vector_db_config=vector_db_config,
            graph_db_config=graph_db_config,
        )


async def run_cognify_blocking(
    tasks,
    user,
    datasets,
    pipeline_name,
    datapoints=None,
    is_stream_info_enabled=False,
    vector_db_config=None,
    graph_db_config=None,
):
    pipeline_run_info = None

    async for run_info in cognee_pipeline(
        tasks=tasks,
        datasets=datasets,
        user=user,
        pipeline_name=pipeline_name,
        datapoints=datapoints,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
    ):
        pipeline_run_info = run_info

        if (
            is_stream_info_enabled
            and not isinstance(pipeline_run_info, PipelineRunStarted)
            and not isinstance(pipeline_run_info, PipelineRunCompleted)
        ):
            pipeline_run_id = get_crewai_pipeline_run_id(user.id)
            pipeline_run_info.payload = await get_formatted_graph_data()
            push_to_queue(pipeline_run_id, pipeline_run_info)

    return pipeline_run_info


async def run_cognify_as_background_process(
    tasks,
    user,
    datasets,
    datapoints,
    pipeline_name,
    vector_db_config,
    graph_db_config,
):
    pipeline_run = cognee_pipeline(
        tasks=tasks,
        user=user,
        datasets=datasets,
        pipeline_name=pipeline_name,
        datapoints=datapoints,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
    )

    pipeline_run_started_info = await anext(pipeline_run)

    async def handle_rest_of_the_run():
        while True:
            try:
                pipeline_run_info = await anext(pipeline_run)

                pipeline_run_info.payload = await get_formatted_graph_data()

                push_to_queue(pipeline_run_info.pipeline_run_id, pipeline_run_info)

                if isinstance(pipeline_run_info, PipelineRunCompleted):
                    break
            except StopAsyncIteration:
                break

    asyncio.create_task(handle_rest_of_the_run())

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
