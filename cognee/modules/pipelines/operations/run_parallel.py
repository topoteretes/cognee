from typing import Any, Callable, Generator, List, Optional
import asyncio
from cognee.shared.logging_utils import get_logger
from ..tasks.task import Task


logger = get_logger("run_parallel")


def run_tasks_parallel(
    tasks: List[Task], task_timeout: Optional[int] = None
) -> Callable[[Any], Generator[Any, Any, Any]]:
    """Execute multiple tasks in parallel with optional timeout and error isolation.
    
    Args:
        tasks: List of tasks to execute in parallel
        task_timeout: Optional timeout in seconds for each task. If None, no timeout is applied.
    
    Returns:
        A Task that executes all tasks in parallel and returns the last result.
    """
    async def parallel_run(*args, **kwargs):
        parallel_tasks = [
            asyncio.create_task(task.run(*args, **kwargs)) for task in tasks
        ]

        # Apply timeout to each task if specified
        if task_timeout is not None:
            wrapped_tasks = [
                asyncio.wait_for(t, timeout=task_timeout) for t in parallel_tasks
            ]
        else:
            wrapped_tasks = parallel_tasks

        # Use return_exceptions=True to isolate failures and prevent cascading cancellations
        results = await asyncio.gather(*wrapped_tasks, return_exceptions=True)

        # Log and handle exceptions gracefully
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, asyncio.TimeoutError):
                logger.error(
                    f"Parallel task {i} timed out after {task_timeout}s"
                )
            elif isinstance(result, asyncio.CancelledError):
                logger.warning(f"Parallel task {i} was cancelled")
            elif isinstance(result, Exception):
                logger.error(f"Parallel task {i} failed: {result}", exc_info=result)
            else:
                successful_results.append(result)

        # Return last successful result, or empty list if all failed
        return successful_results[-1] if successful_results else []

    return Task(parallel_run)
