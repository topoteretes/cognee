import asyncio
from collections.abc import Callable, Generator
from typing import Any

from ..tasks.task import Task


def run_tasks_parallel(tasks: list[Task]) -> Callable[[Any], Generator[Any, Any, Any]]:
    async def parallel_run(*args, **kwargs):
        parallel_tasks = [asyncio.create_task(task.run(*args, **kwargs)) for task in tasks]

        results = await asyncio.gather(*parallel_tasks)
        return results[-1] if results else []

    return Task(parallel_run)
