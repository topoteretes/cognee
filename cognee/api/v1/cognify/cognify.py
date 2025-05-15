import asyncio
from cognee.shared.logging_utils import get_logger
from typing import Union, Optional
from pydantic import BaseModel

from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.documents import (
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.pipelines import cognee_pipeline

logger = get_logger("cognify")

update_status_lock = asyncio.Lock()


async def cognify(
    datasets: Union[str, list[str]] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
):
    tasks = await get_default_tasks(user, graph_model, chunker, chunk_size, ontology_file_path)

    return await cognee_pipeline(
        tasks=tasks, datasets=datasets, user=user, pipeline_name="cognify_pipeline"
    )


async def get_default_tasks(  # TODO: Find out a better way to do this (Boris's comment)
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
) -> list[Task]:
    default_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_documents, user=user, permissions=["write"]),
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
