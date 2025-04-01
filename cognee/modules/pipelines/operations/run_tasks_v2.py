import json
from typing import Any
from collections import deque
from uuid import UUID, NAMESPACE_OID, uuid4, uuid5

from cognee.shared.logging_utils import get_logger

from cognee.modules.pipelines.operations import (
    log_pipeline_run_start,
    log_pipeline_run_complete,
    log_pipeline_run_error,
)
from cognee.shared.utils import send_telemetry
from cognee.modules.settings import get_current_settings
from cognee.modules.users.methods import get_default_user

from .input_output import get_input_results, get_input_tasks
from ..tasks.Task import Task
from ..exceptions import WrongTaskOrderException

logger = get_logger("run_tasks(tasks: [Task], data)")


async def run_tasks_base(tasks: list[Task], data=None, user=None):
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
        task_graph[task.executable] = get_input_tasks(task.task_config.inputs)
        for dependendent_task in task_graph[task.executable]:
            dependents.setdefault(dependendent_task, []).append(task.executable)

    # Find tasks without dependencies
    ready_queue = deque([task for task in tasks if not task_graph[task.executable]])

    # Execute tasks in order
    while ready_queue:
        task = ready_queue.popleft()
        task_inputs = (
            get_input_results(results, task) if task.task_config.inputs else pipeline_input
        )

        async for task_execution_info in task.run(*task_inputs):  # Run task and store result
            if not task_execution_info.is_done:
                results[task.executable] = task_execution_info.result
            elif task_execution_info.is_done and task.executable not in results:
                results[task.executable] = task_execution_info.result

            if task_execution_info.is_done:
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


async def run_tasks_with_telemetry(tasks: list[Task], data, pipeline_name: str):
    config = get_current_settings()

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

    user = await get_default_user()

    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Started",
            user.id,
            additional_properties={
                "pipeline_name": pipeline_name,
            }
            | config,
        )

        async for result in run_tasks_base(tasks, data, user):
            yield result

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Completed",
            user.id,
            additional_properties={
                "pipeline_name": pipeline_name,
            },
        )
    except Exception as error:
        logger.error(
            "Pipeline run errored: `%s`\n%s\n",
            pipeline_name,
            str(error),
            exc_info=True,
        )
        send_telemetry(
            "Pipeline Run Errored",
            user.id,
            additional_properties={
                "pipeline_name": pipeline_name,
            }
            | config,
        )

        raise error


async def run_tasks(
    tasks: list[Task],
    dataset_id: UUID = uuid4(),
    data: Any = None,
    pipeline_name: str = "unknown_pipeline",
):
    pipeline_id = uuid5(NAMESPACE_OID, pipeline_name)

    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset_id, data)

    yield pipeline_run

    pipeline_run_id = pipeline_run.pipeline_run_id

    try:
        async for _ in run_tasks_with_telemetry(tasks, data, pipeline_name):
            pass

        yield await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data
        )

    except Exception as e:
        yield await log_pipeline_run_error(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data, e
        )
        raise e
