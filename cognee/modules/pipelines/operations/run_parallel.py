from typing import Any, Callable, Generator, List
import asyncio
from ..tasks.task import Task


def run_tasks_parallel(tasks: List[Task]) -> Callable[[Any], Generator[Any, Any, Any]]:
    async def parallel_run(*args, **kwargs):
        parallel_tasks = [asyncio.create_task(task.run(*args, **kwargs)) for task in tasks]

        results = await asyncio.gather(*parallel_tasks)
        return results[len(results) - 1] if len(results) > 1 else []

    return Task(parallel_run)
