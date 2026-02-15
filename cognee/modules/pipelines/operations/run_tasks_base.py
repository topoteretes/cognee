import inspect
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version
from cognee.modules.observability.trace_context import is_tracing_enabled

from cognee.infrastructure.engine import DataPoint
from ..tasks.task import Task

logger = get_logger("run_tasks_base")


def _get_tracer():
    if is_tracing_enabled():
        from cognee.modules.observability.tracing import get_tracer

        return get_tracer()
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

    tracer = _get_tracer()
    task_name = running_task.executable.__name__

    if tracer is not None:
        from cognee.modules.observability.tracing import COGNEE_PIPELINE_TASK_NAME

        span_ctx = tracer.start_as_current_span(f"cognee.task.{task_name}")
    else:
        from contextlib import nullcontext

        span_ctx = nullcontext()

    with span_ctx as span:
        if span is not None:
            span.set_attribute(COGNEE_PIPELINE_TASK_NAME, task_name)

        try:
            async for result_data in running_task.execute(args, next_task_batch_size):
                async for result in run_tasks_base(leftover_tasks, result_data, user, context):
                    yield result

            logger.info(f"{task_type} task completed: `{task_name}`")
            send_telemetry(
                f"{task_type} Task Completed",
                user_id=user.id,
                additional_properties={
                    "task_name": task_name,
                    "cognee_version": cognee_version,
                    "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
                },
            )
        except Exception as error:
            if span is not None:
                from opentelemetry.trace import StatusCode

                span.set_status(StatusCode.ERROR, str(error))
                span.record_exception(error)
            logger.error(
                f"{task_type} task errored: `{task_name}`\n{str(error)}\n",
                exc_info=True,
            )
            send_telemetry(
                f"{task_type} Task Errored",
                user_id=user.id,
                additional_properties={
                    "task_name": task_name,
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
