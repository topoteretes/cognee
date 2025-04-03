from typing import List
from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.pipelines.tasks import TaskConfig
from cognee.tasks.documents import (
    classify_documents,
    check_permissions_on_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.pipelines import run_tasks, merge_needs
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.infrastructure.llm import get_max_chunk_tokens


async def get_default_tasks_by_indices(
    indices: List[int], chunk_size: int = None, chunker=TextChunker
) -> List[Task]:
    """Returns default tasks filtered by the provided indices."""
    all_tasks = await get_default_tasks(chunker=chunker, chunk_size=chunk_size)

    if any(i < 0 or i >= len(all_tasks) for i in indices):
        raise IndexError(
            f"Task indices {indices} out of range. Valid range: 0-{len(all_tasks) - 1}"
        )

    return [all_tasks[i] for i in indices]


async def get_no_summary_tasks(
    chunk_size: int = None,
    chunker=TextChunker,
    user=None,
    graph_model=KnowledgeGraph,
    ontology_file_path=None,
) -> List[Task]:
    """Returns default tasks without summarization tasks."""
    if user is None:
        user = await get_default_user()

    ontology_adapter = OntologyResolver(ontology_file=ontology_file_path)

    tasks = [
        Task(classify_documents),
        Task(
            check_permissions_on_documents,
            user=user,
            permissions=["write"],
            task_config=TaskConfig(needs=[classify_documents]),
        ),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
            task_config=TaskConfig(needs=[check_permissions_on_documents], output_batch_size=10),
        ),
        Task(
            extract_graph_from_data,
            graph_model=graph_model,
            ontology_adapter=ontology_adapter,
            task_config=TaskConfig(needs=[extract_chunks_from_documents]),
        ),
        Task(
            add_data_points,
            task_config=TaskConfig(needs=[extract_graph_from_data]),
        ),
    ]

    return tasks


async def get_just_chunks_tasks(
    chunk_size: int = None, chunker=TextChunker, user=None
) -> List[Task]:
    """Returns default tasks with only chunk extraction and data points addition."""
    if user is None:
        user = await get_default_user()

    tasks = [
        Task(classify_documents),
        Task(
            check_permissions_on_documents,
            user=user,
            permissions=["write"],
            task_config=TaskConfig(needs=[classify_documents]),
        ),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
            task_config=TaskConfig(needs=[check_permissions_on_documents], output_batch_size=10),
        ),
        Task(
            add_data_points,
            task_config=TaskConfig(needs=[extract_chunks_from_documents]),
        ),
    ]

    return tasks
