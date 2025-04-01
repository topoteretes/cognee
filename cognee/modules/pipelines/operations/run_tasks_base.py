from collections import deque

from cognee.shared.logging_utils import get_logger

from .needs import get_need_task_results, get_task_needs
from ..tasks.Task import Task, TaskExecutionCompleted, TaskExecutionInfo
from ..exceptions import WrongTaskOrderException

logger = get_logger("run_tasks(tasks: [Task], data)")


async def run_tasks_base(tasks: list[Task], data=None, context=None):
    if len(tasks) == 0:
        return

    pipeline_input = [data] if data is not None else []

    """Run tasks in dependency order and return results."""
    task_graph = {}  # Map task to its dependencies
    dependents = {}  # Reverse dependencies (who depends on whom)
    results = {}
    number_of_executed_tasks = 0

    tasks_map = {task.executable: task for task in tasks}  # Map task executable to task object

    # Build task dependency graph
    for task in tasks:
        task_graph[task.executable] = get_task_needs(task.task_config.needs)
        for dependendent_task in task_graph[task.executable]:
            dependents.setdefault(dependendent_task, []).append(task.executable)

    # Find tasks without dependencies
    ready_queue = deque([task for task in tasks if not task_graph[task.executable]])

    # Execute tasks in order
    while ready_queue:
        task = ready_queue.popleft()
        task_inputs = (
            get_need_task_results(results, task) if task.task_config.needs else pipeline_input
        )

        async for task_execution_info in task.run(*task_inputs):  # Run task and store result
            if isinstance(task_execution_info, TaskExecutionInfo):  # Update result as it comes
                results[task.executable] = task_execution_info.result

            if isinstance(task_execution_info, TaskExecutionCompleted):
                if task.executable not in results:  # If result not already set, set it
                    results[task.executable] = task_execution_info.result

                number_of_executed_tasks += 1

            yield task_execution_info

        # Process tasks depending on this task
        for dependendent_task in dependents.get(task.executable, []):
            task_graph[dependendent_task].remove(task.executable)  # Mark dependency as resolved
            if not task_graph[dependendent_task]:  # If all dependencies resolved, add to queue
                ready_queue.append(tasks_map[dependendent_task])

    if number_of_executed_tasks != len(tasks):
        raise WrongTaskOrderException(
            f"{number_of_executed_tasks}/{len(tasks)} tasks executed. You likely have some disconneted tasks or circular dependency."
        )
