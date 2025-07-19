import inspect
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry
from cognee.exceptions import (
    PipelineExecutionError,
    CogneeTransientError,
    CogneeSystemError,
    CogneeUserError,
    LLMConnectionError,
    DatabaseConnectionError,
)

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
    """
    Handle common task workflow with enhanced error handling and recovery strategies.

    This function provides comprehensive error handling for pipeline tasks with:
    - Context-aware error reporting
    - Automatic retry for transient errors
    - Detailed error logging and telemetry
    - User-friendly error messages
    """
    task_type = running_task.task_type
    task_name = running_task.executable.__name__

    logger.info(
        f"{task_type} task started: `{task_name}`",
        extra={
            "task_type": task_type,
            "task_name": task_name,
            "user_id": user.id,
            "context": context,
        },
    )

    send_telemetry(
        f"{task_type} Task Started",
        user_id=user.id,
        additional_properties={
            "task_name": task_name,
        },
    )

    has_context = any(
        [key == "context" for key in inspect.signature(running_task.executable).parameters.keys()]
    )

    if has_context:
        args.append(context)

    # Retry configuration for transient errors
    max_retries = 3
    retry_count = 0

    while retry_count <= max_retries:
        try:
            async for result_data in running_task.execute(args, next_task_batch_size):
                async for result in run_tasks_base(leftover_tasks, result_data, user, context):
                    yield result

            logger.info(
                f"{task_type} task completed: `{task_name}`",
                extra={
                    "task_type": task_type,
                    "task_name": task_name,
                    "user_id": user.id,
                    "retry_count": retry_count,
                },
            )

            send_telemetry(
                f"{task_type} Task Completed",
                user_id=user.id,
                additional_properties={
                    "task_name": task_name,
                    "retry_count": retry_count,
                },
            )
            return  # Success, exit retry loop

        except CogneeTransientError as error:
            retry_count += 1
            if retry_count <= max_retries:
                logger.warning(
                    f"Transient error in {task_type} task `{task_name}`, retrying ({retry_count}/{max_retries}): {error}",
                    extra={
                        "task_type": task_type,
                        "task_name": task_name,
                        "user_id": user.id,
                        "retry_count": retry_count,
                        "error_type": error.__class__.__name__,
                    },
                )
                # Exponential backoff for retries
                import asyncio

                await asyncio.sleep(2**retry_count)
                continue
            else:
                # Max retries exceeded, raise enhanced error
                raise PipelineExecutionError(
                    pipeline_name=f"{task_type}_pipeline",
                    task_name=task_name,
                    error_details=f"Max retries ({max_retries}) exceeded for transient error: {error}",
                )

        except (CogneeUserError, CogneeSystemError) as error:
            # These errors shouldn't be retried, re-raise as pipeline execution error
            logger.error(
                f"{task_type} task failed: `{task_name}` - {error.__class__.__name__}: {error}",
                extra={
                    "task_type": task_type,
                    "task_name": task_name,
                    "user_id": user.id,
                    "error_type": error.__class__.__name__,
                    "error_context": getattr(error, "context", {}),
                },
                exc_info=True,
            )

            send_telemetry(
                f"{task_type} Task Errored",
                user_id=user.id,
                additional_properties={
                    "task_name": task_name,
                    "error_type": error.__class__.__name__,
                },
            )

            # Wrap in pipeline execution error with additional context
            raise PipelineExecutionError(
                pipeline_name=f"{task_type}_pipeline",
                task_name=task_name,
                error_details=f"{error.__class__.__name__}: {error}",
                context={
                    "original_error": error.__class__.__name__,
                    "original_context": getattr(error, "context", {}),
                    "user_id": user.id,
                    "task_args": str(args)[:200],  # Truncate for logging
                },
            )

        except Exception as error:
            # Unexpected error, wrap in enhanced exception
            logger.error(
                f"{task_type} task encountered unexpected error: `{task_name}` - {error}",
                extra={
                    "task_type": task_type,
                    "task_name": task_name,
                    "user_id": user.id,
                    "error_type": error.__class__.__name__,
                },
                exc_info=True,
            )

            send_telemetry(
                f"{task_type} Task Errored",
                user_id=user.id,
                additional_properties={
                    "task_name": task_name,
                    "error_type": error.__class__.__name__,
                },
            )

            # Check if this might be a known error type we can categorize
            error_message = str(error).lower()
            if any(term in error_message for term in ["connection", "timeout", "network"]):
                if (
                    "llm" in error_message
                    or "openai" in error_message
                    or "anthropic" in error_message
                ):
                    raise LLMConnectionError(provider="Unknown", model="Unknown", reason=str(error))
                elif "database" in error_message or "sql" in error_message:
                    raise DatabaseConnectionError(db_type="Unknown", reason=str(error))

            # Default to pipeline execution error
            raise PipelineExecutionError(
                pipeline_name=f"{task_type}_pipeline",
                task_name=task_name,
                error_details=f"Unexpected error: {error}",
                context={
                    "error_type": error.__class__.__name__,
                    "user_id": user.id,
                    "task_args": str(args)[:200],  # Truncate for logging
                },
            )


async def run_tasks_base(tasks: list[Task], data=None, user: User = None, context: dict = None):
    """
    Base function to execute tasks in a pipeline with enhanced error handling.

    Provides comprehensive error handling, logging, and recovery strategies for pipeline execution.
    """
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
