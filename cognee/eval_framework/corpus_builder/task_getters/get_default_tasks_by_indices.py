from typing import List
from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.chunking.TextChunker import TextChunker


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


async def get_no_summary_tasks(chunk_size: int = None, chunker=TextChunker) -> List[Task]:
    """Returns default tasks without summarization tasks."""
    # Default tasks indices: 0=classify, 1=check_permissions, 2=extract_chunks, 3=extract_graph, 4=summarize, 5=add_data_points
    return await get_default_tasks_by_indices(
        [0, 1, 2, 3, 5], chunk_size=chunk_size, chunker=chunker
    )


async def get_just_chunks_tasks(chunk_size: int = None, chunker=TextChunker) -> List[Task]:
    """Returns default tasks with only chunk extraction and data points addition."""
    # Default tasks indices: 0=classify, 1=check_permissions, 2=extract_chunks, 3=extract_graph, 4=summarize, 5=add_data_points
    return await get_default_tasks_by_indices([0, 1, 2, 5], chunk_size=chunk_size, chunker=chunker)
