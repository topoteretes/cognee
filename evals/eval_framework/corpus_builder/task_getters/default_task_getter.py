from cognee.api.v1.cognify.cognify_v2 import get_default_tasks
from typing import List
from evals.eval_framework.corpus_builder.task_getters.base_task_getter import BaseTaskGetter
from cognee.modules.pipelines.tasks.Task import Task


class DefaultTaskGetter(BaseTaskGetter):
    """Default task getter that retrieves tasks using the standard get_default_tasks function."""

    async def get_tasks(self) -> List[Task]:
        """Retrieve default tasks asynchronously."""
        return await get_default_tasks()
