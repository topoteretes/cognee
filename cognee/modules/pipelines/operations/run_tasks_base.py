import inspect
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

from cognee.infrastructure.engine import DataPoint
from ..tasks.task import Task

logger = get_logger("run_tasks_base")


def _stamp_provenance(data, pipeline_name, task_name, visited=None, note_set=None, user_label=None):
    """Recursively stamp DataPoints with provenance. Only sets if currently None."""
    if visited is None:
        visited = set()

    if isinstance(data, DataPoint):
        obj_id = id(data)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if data.source_pipeline is None:
            data.source_pipeline = pipeline_name
        if data.source_task is None:
            data.source_task = task_name
        if data.source_user is None and user_label is not None:
            data.source_user = user_label

        # Propagate note_set from parent or pick up from this data point
        current_note_set = note_set
        if data.source_node_set is not None:
            current_note_set = data.source_node_set
        elif current_note_set is not None and data.source_node_set is None:
            data.source_node_set = current_note_set

        # Recurse into DataPoint model fields to stamp nested DataPoints
        for field_name in data.model_fields:
            field_value = getattr(data, field_name, None)
            if field_value is not None:
                _stamp_provenance(
                    field_value,
                    pipeline_name,
                    task_name,
                    visited,
                    current_note_set,
                    user_label,
                )

    elif isinstance(data, (list, tuple)):
        for item in data:
            _stamp_provenance(item, pipeline_name, task_name, visited, note_set, user_label)


def _extract_note_set(args):
    """Extract source_node_set from input args to propagate across task boundaries."""
    for arg in args:
        if isinstance(arg, DataPoint) and arg.source_node_set is not None:
            return arg.source_node_set
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, DataPoint) and item.source_node_set is not None:
                    return item.source_node_set
    return None


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

    kwargs = {}

    if has_context:
        kwargs["context"] = context

    try:
        task_name = running_task.executable.__name__
        pipe_name = context.get("pipeline_name") if isinstance(context, dict) else None
        input_note_set = _extract_note_set(args)
        user_label = getattr(user, "email", None) or (str(user.id) if user else None)

        async for result_data in running_task.execute(args, kwargs, next_task_batch_size):
            _stamp_provenance(
                result_data,
                pipe_name,
                task_name,
                note_set=input_note_set,
                user_label=user_label,
            )
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
