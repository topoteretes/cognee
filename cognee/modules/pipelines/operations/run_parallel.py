from typing import Any, Callable, Generator
import asyncio
from ..tasks.Task import Task

def run_tasks_parallel(tasks: [Task]) -> Callable[[Any], Generator[Any, Any, Any]]:
    async def parallel_run(*args, **kwargs):
        parallel_tasks = [asyncio.create_task(task.run(*args, **kwargs)) for task in tasks]

        results = await asyncio.gather(*parallel_tasks)
        return results[len(results) - 1] if len(results) > 1 else []

    return Task(parallel_run)
