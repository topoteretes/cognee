import inspect
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

from ..tasks.task import Task

logger = get_logger("run_tasks_base")


async def handle_task(
    running_task: Task,
    args: list,
    leftover_tasks: list[Task],
    next_task_batch_size: int,
    user: User,
    context: dict = None,
):
    """Handle common task workflow with logging, telemetry, and error handling around the core execution logic."""
    task_type = running_task.task_type

    logger.info(f"{task_type} task started: `{running_task.executable.__name__}`")
    send_telemetry(
        f"{task_type} Task Started",
        user_id=user.id,
        additional_properties={
            "task_name": running_task.executable.__name__,
            "cognee_version": cognee_version,
            "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
        },
    )

    has_context = any(
        [key == "context" for key in inspect.signature(running_task.executable).parameters.keys()]
    )

    if has_context:
        args.append(context)

    try:
        async for result_data in running_task.execute(args, next_task_batch_size):
            async for result in run_tasks_base(leftover_tasks, result_data, user, context):
                yield result

        logger.info(f"{task_type} task completed: `{running_task.executable.__name__}`")
        send_telemetry(
            f"{task_type} Task Completed",
            user_id=user.id,
            additional_properties={
                "task_name": running_task.executable.__name__,
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
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
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            },
        )
        raise error


async def run_tasks_base(tasks: list[Task], data=None, user: User = None, context: dict = None):
    """Base function to execute tasks in a pipeline, handling task type detection and execution."""
    if len(tasks) == 0:
        yield data
        return

    args = [data] if data is not None else []

    running_task = tasks[0]
    leftover_tasks = tasks[1:]
    next_task = leftover_tasks[0] if len(leftover_tasks) > 0 else None
    next_task_batch_size = next_task.task_config["batch_size"] if next_task else 1

    async for result in handle_task(
        running_task, args, leftover_tasks, next_task_batch_size, user, context
    ):
        yield result
