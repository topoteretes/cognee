from typing import Union, Optional, List
from uuid import UUID
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm import get_max_chunk_tokens, get_llm_config

from cognee.api.v1.cognify.cognify import run_cognify_blocking
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver

from cognee.modules.users.models import User
from cognee.tasks.documents import (
    check_permissions_on_dataset,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.temporal_poc.event_extraction import extract_events_and_entities
from cognee.temporal_poc.event_knowledge_graph import process_event_knowledge_graph

logger = get_logger("temporal_cognify")


async def get_temporal_tasks(
    user: User = None, chunker=TextChunker, chunk_size: int = None
) -> list[Task]:
    temporal_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_dataset, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),
        Task(extract_events_and_entities, task_config={"chunk_size": 10}),
        Task(process_event_knowledge_graph),
        Task(add_data_points, task_config={"batch_size": 10}),
    ]

    return temporal_tasks


async def temporal_cognify(
    datasets: Union[str, list[str], list[UUID]] = None,
    user: User = None,
    chunker=TextChunker,
    chunk_size: int = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    incremental_loading: bool = True,
):
    tasks = await get_temporal_tasks(user, chunker, chunk_size)

    return await run_cognify_blocking(
        tasks=tasks,
        user=user,
        datasets=datasets,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=incremental_loading,
    )
