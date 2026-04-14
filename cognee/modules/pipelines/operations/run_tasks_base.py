from typing import Optional

from cognee.modules.observability import OtelStatusCode as StatusCode
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.observability import (
    new_span,
    COGNEE_PIPELINE_TASK_NAME,
    COGNEE_RESULT_SUMMARY,
    COGNEE_RESULT_COUNT,
)
from cognee.infrastructure.engine import DataPoint
from ..tasks.task import Task

logger = get_logger("run_tasks_base")


def _build_result_summary(executable, task_name: str, count: int) -> str:
    """Build a human-readable result summary for a completed task.

    Reads the ``__task_summary__`` attribute set by the ``@task_summary``
    decorator.  Falls back to a generic message when no template is defined.
    """
    template = getattr(executable, "__task_summary__", None)
    if template:
        return template.format(n=count)
    return f"{task_name} produced {count} result(s)"


def _stamp_provenance(
    data, pipeline_name, task_name, visited=None, node_set=None, user_label=None, content_hash=None
):
    """Recursively stamp DataPoints with provenance. Only sets if currently None.

    The ``visited`` set should be persisted across task calls (via
    PipelineContext._provenance_visited) so that DataPoints stamped in
    earlier stages are skipped in later ones.
    """
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

        # Propagate node_set from parent or pick up from this data point
        current_node_set = node_set
        if data.source_node_set is not None:
            current_node_set = data.source_node_set
        elif current_node_set is not None and data.source_node_set is None:
            data.source_node_set = current_node_set

        # Propagate content_hash from parent or pick up from this data point
        current_hash = content_hash
        if data.source_content_hash is not None:
            current_hash = data.source_content_hash
        elif current_hash is not None and data.source_content_hash is None:
            data.source_content_hash = current_hash

        # Recurse into DataPoint model fields to stamp nested DataPoints
        for field_name in data.model_fields:
            field_value = getattr(data, field_name, None)
            if field_value is not None:
                _stamp_provenance(
                    field_value,
                    pipeline_name,
                    task_name,
                    visited,
                    current_node_set,
                    user_label,
                    current_hash,
                )

    elif isinstance(data, (list, tuple)):
        for item in data:
            _stamp_provenance(
                item, pipeline_name, task_name, visited, node_set, user_label, content_hash
            )


def _extract_node_set(args):
    """Extract source_node_set from input args to propagate across task boundaries."""
    for arg in args:
        if isinstance(arg, DataPoint) and arg.source_node_set is not None:
            return arg.source_node_set
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, DataPoint) and item.source_node_set is not None:
                    return item.source_node_set
    return None


def _extract_content_hash(args):
    """Extract content_hash from input Data items to propagate to output DataPoints."""
    from cognee.modules.data.models.Data import Data

    for arg in args:
        if isinstance(arg, Data) and arg.content_hash is not None:
            return arg.content_hash
        if isinstance(arg, DataPoint) and arg.source_content_hash is not None:
            return arg.source_content_hash
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, Data) and item.content_hash is not None:
                    return item.content_hash
                if isinstance(item, DataPoint) and item.source_content_hash is not None:
                    return item.source_content_hash
    return None


async def handle_task(
    running_task: Task,
    args: list,
    leftover_tasks: list[Task],
    next_task_batch_size: int,
    user: User,
    ctx: Optional[PipelineContext] = None,
):
    """Handle common task workflow with logging, telemetry, and error handling."""
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

    # Pass ctx only to tasks that declare it in their signature.
    # Task caches this check as accepts_ctx at construction time.
    kwargs = {}
    if ctx is not None and running_task.accepts_ctx:
        kwargs["ctx"] = ctx

    task_name = running_task.executable.__name__

    with new_span(f"cognee.pipeline.task.{task_name}") as span:
        span.set_attribute(COGNEE_PIPELINE_TASK_NAME, task_name)

        try:
            result_count = 0
            pipe_name = ctx.pipeline_name if ctx else None
            input_node_set = _extract_node_set(args)
            input_content_hash = _extract_content_hash(args)
            user_label = getattr(user, "email", None) or (str(user.id) if user else None)
            # Reuse the visited set across tasks so already-stamped
            # DataPoints are skipped in subsequent pipeline stages.
            provenance_visited = ctx._provenance_visited if ctx else None

            async for result_data in running_task.execute(args, kwargs, next_task_batch_size):
                if isinstance(result_data, list):
                    result_count += len(result_data)
                else:
                    result_count += 1

                _stamp_provenance(
                    result_data,
                    pipe_name,
                    task_name,
                    visited=provenance_visited,
                    node_set=input_node_set,
                    user_label=user_label,
                    content_hash=input_content_hash,
                )

                async for result in run_tasks_base(leftover_tasks, result_data, user, ctx):
                    yield result

            span.set_attribute(COGNEE_RESULT_COUNT, result_count)
            span.set_attribute(
                COGNEE_RESULT_SUMMARY,
                _build_result_summary(running_task.executable, task_name, result_count),
            )

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


async def run_tasks_base(
    tasks: list[Task],
    data=None,
    user: User = None,
    ctx: Optional[PipelineContext] = None,
):
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
        running_task, args, leftover_tasks, next_task_batch_size, user, ctx
    ):
        yield result
