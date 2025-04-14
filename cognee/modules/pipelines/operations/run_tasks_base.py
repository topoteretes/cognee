import inspect
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry

from ..tasks.Task import Task

logger = get_logger("run_tasks_base")


async def execute_async_generator_task(
    running_task: Task,
    args: list,
    leftover_tasks: list[Task],
    next_task_batch_size: int,
    user: User,
):
    """Execute async generator task and process results with batching."""
    results = []

    async_iterator = running_task.run(*args)

    async for partial_result in async_iterator:
        results.append(partial_result)

        if len(results) == next_task_batch_size:
            async for result in run_tasks_base(
                leftover_tasks,
                results,
                user=user,
            ):
                yield result

            results = []

    if len(results) > 0:
        async for result in run_tasks_base(leftover_tasks, results, user):
            yield result

        results = []


async def execute_generator_task(
    running_task: Task,
    args: list,
    leftover_tasks: list[Task],
    next_task_batch_size: int,
    user: User,
):
    """Execute generator task and process results with batching."""
    results = []

    for partial_result in running_task.run(*args):
        results.append(partial_result)

        if len(results) == next_task_batch_size:
            async for result in run_tasks_base(leftover_tasks, results, user):
                yield result

            results = []

    if len(results) > 0:
        async for result in run_tasks_base(leftover_tasks, results, user):
            yield result

        results = []


async def execute_coroutine_task(
    running_task: Task, args: list, leftover_tasks: list[Task], user: User
):
    """Execute coroutine task and process single result."""
    task_result = await running_task.run(*args)

    async for result in run_tasks_base(leftover_tasks, task_result, user):
        yield result


async def execute_function_task(
    running_task: Task, args: list, leftover_tasks: list[Task], user: User
):
    """Execute function task and process single result."""
    task_result = running_task.run(*args)

    async for result in run_tasks_base(leftover_tasks, task_result, user):
        yield result


def get_task_type(running_task: Task):
    """Determine the type of task based on the executable."""
    if inspect.isasyncgenfunction(running_task.executable):
        return "Async Generator"
    elif inspect.isgeneratorfunction(running_task.executable):
        return "Generator"
    elif inspect.iscoroutinefunction(running_task.executable):
        return "Coroutine"
    elif inspect.isfunction(running_task.executable):
        return "Function"
    else:
        raise ValueError(f"Unsupported task type: {running_task.executable}")


def get_task_executor(task_type: str):
    """Get the appropriate executor function based on task type."""
    if task_type == "Async Generator":
        return execute_async_generator_task
    elif task_type == "Generator":
        return execute_generator_task
    elif task_type == "Coroutine":
        return execute_coroutine_task
    elif task_type == "Function":
        return execute_function_task
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


async def handle_task(
    running_task: Task,
    args: list,
    leftover_tasks: list[Task],
    next_task_batch_size: int,
    user: User,
):
    """Handle common task workflow with logging, telemetry, and error handling around the core execution logic."""
    # Get task information using the helper functions
    task_type = get_task_type(running_task)
    executor = get_task_executor(task_type)

    # Determine executor args based on task type
    execute_args = (args, leftover_tasks)
    if task_type in ["Async Generator", "Generator"]:
        execute_args += (next_task_batch_size,)

    logger.info(f"{task_type} task started: `{running_task.executable.__name__}`")
    send_telemetry(
        f"{task_type} Task Started",
        user_id=user.id,
        additional_properties={
            "task_name": running_task.executable.__name__,
        },
    )
    try:
        # Add user to the execute args
        complete_args = execute_args + (user,)
        async for result in executor(running_task, *complete_args):
            yield result

        logger.info(f"{task_type} task completed: `{running_task.executable.__name__}`")
        send_telemetry(
            f"{task_type} Task Completed",
            user_id=user.id,
            additional_properties={
                "task_name": running_task.executable.__name__,
            },
        )
    except Exception as error:
        logger.error(
            f"{task_type} task errored: `{running_task.executable.__name__}`\n{str(error)}\n",
            exc_info=True,
        )
        send_telemetry(
            f"{task_type} Task Errored",
            user_id=user.id,
            additional_properties={
                "task_name": running_task.executable.__name__,
            },
        )
        raise error


async def run_tasks_base(tasks: list[Task], data=None, user: User = None):
    """Base function to execute tasks in a pipeline, handling task type detection and execution."""
    if len(tasks) == 0:
        yield data
        return

    args = [data] if data is not None else []

    running_task = tasks[0]
    leftover_tasks = tasks[1:]
    next_task = leftover_tasks[0] if len(leftover_tasks) > 0 else None
    next_task_batch_size = next_task.task_config["batch_size"] if next_task else 1

    # Execute with the common handler that determines and runs the appropriate task
    async for result in handle_task(running_task, args, leftover_tasks, next_task_batch_size, user):
        yield result
