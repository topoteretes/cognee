from typing import List
from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver


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
    # Get base tasks (0=classify, 1=check_permissions, 2=extract_chunks)
    base_tasks = await get_default_tasks_by_indices([0, 1, 2], chunk_size, chunker)

    ontology_adapter = OntologyResolver(ontology_file=ontology_file_path)

    graph_task = Task(
        extract_graph_from_data,
        graph_model=graph_model,
        ontology_adapter=ontology_adapter,
        task_config={"batch_size": 10},
    )

    add_data_points_task = Task(add_data_points, task_config={"batch_size": 10})

    return base_tasks + [graph_task, add_data_points_task]


async def get_just_chunks_tasks(
    chunk_size: int = None, chunker=TextChunker, user=None
) -> List[Task]:
    """Returns default tasks with only chunk extraction and data points addition."""
    # Get base tasks (0=classify, 1=check_permissions, 2=extract_chunks)
    base_tasks = await get_default_tasks_by_indices([0, 1, 2], chunk_size, chunker)

    add_data_points_task = Task(add_data_points, task_config={"batch_size": 10})

    return base_tasks + [add_data_points_task]
