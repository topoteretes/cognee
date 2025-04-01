import json
from typing import Any
from uuid import UUID, NAMESPACE_OID, uuid4, uuid5

from cognee.modules.pipelines.operations import (
    log_pipeline_run_start,
    log_pipeline_run_complete,
    log_pipeline_run_error,
)
from cognee.modules.users.methods import get_default_user
from cognee.modules.settings import get_current_settings
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger

from ..tasks.Task import Task, TaskExecutionCompleted, TaskExecutionErrored, TaskExecutionStarted
from .run_tasks_base import run_tasks_base

logger = get_logger("run_tasks(tasks: [Task], data)")


async def run_tasks_with_telemetry(
    tasks: list[Task], data, pipeline_name: str, context: dict = None
):
    config = get_current_settings()

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

    user = await get_default_user()

    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Started",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
            }
            | config,
        )

        async for run_task_info in run_tasks_base(tasks, data, context):
            if isinstance(run_task_info, TaskExecutionStarted):
                send_telemetry(
                    "Task Run Started",
                    user.id,
                    additional_properties={
                        "task_name": run_task_info.task.__name__,
                    }
                    | config,
                )

            if isinstance(run_task_info, TaskExecutionCompleted):
                send_telemetry(
                    "Task Run Completed",
                    user.id,
                    additional_properties={
                        "task_name": run_task_info.task.__name__,
                    }
                    | config,
                )

            if isinstance(run_task_info, TaskExecutionErrored):
                send_telemetry(
                    "Task Run Errored",
                    user.id,
                    additional_properties={
                        "task_name": run_task_info.task.__name__,
                        "error": str(run_task_info.error),
                    }
                    | config,
                )
                logger.error(
                    "Task run errored: `%s`\n%s\n",
                    run_task_info.task.__name__,
                    str(run_task_info.error),
                    exc_info=True,
                )

            yield run_task_info

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Completed",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
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
                "pipeline_name": str(pipeline_name),
            }
            | config,
        )

        raise error


async def run_tasks(
    tasks: list[Task],
    dataset_id: UUID = uuid4(),
    data: Any = None,
    pipeline_name: str = "unknown_pipeline",
    context: dict = None,
):
    pipeline_id = uuid5(NAMESPACE_OID, pipeline_name)

    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset_id, data)

    yield pipeline_run

    pipeline_run_id = pipeline_run.pipeline_run_id

    try:
        async for _ in run_tasks_with_telemetry(tasks, data, pipeline_id, context):
            pass

        yield await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data
        )

    except Exception as e:
        yield await log_pipeline_run_error(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data, e
        )
        raise e
