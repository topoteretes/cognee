from cognee.api.v1.cognify.cognify_v2 import get_default_tasks
from typing import List
from cognee.eval_framework.corpus_builder.task_getters.base_task_getter import BaseTaskGetter
from cognee.modules.pipelines.tasks.Task import Task
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.chunking.TextChunker import TextChunker


class DefaultTaskGetter(BaseTaskGetter):
    """Default task getter that retrieves tasks using the standard get_default_tasks function."""

    async def get_tasks(self, chunk_size=1024, chunker=TextChunker) -> List[Task]:
        """Retrieve default tasks asynchronously."""
        return await get_default_tasks(chunk_size=chunk_size, chunker=chunker)
