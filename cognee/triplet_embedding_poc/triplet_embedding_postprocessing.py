from typing import Any

from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.pipelines.tasks.task import Task


logger = get_logger("triplet_embedding_poc")


async def get_triplets_from_graph_store(data) -> Any:
    for i in range(0, 5):
        yield i


async def add_triplets_to_collection(data) -> None:
    print(data)


async def get_triplet_embedding_tasks() -> list[Task]:
    triplet_embedding_tasks = [
        Task(get_triplets_from_graph_store),
        Task(add_triplets_to_collection),
    ]

    return triplet_embedding_tasks


async def triplet_embedding_postprocessing():
    tasks = await get_triplet_embedding_tasks()

    async for result in run_tasks_base(tasks, user=await get_default_user(), data=[]):
        print(result)
