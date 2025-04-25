import json
from cognee.shared.logging_utils import get_logger
from uuid import UUID, uuid4

from typing import Any
from cognee.modules.pipelines.operations import (
    log_pipeline_run_start,
    log_pipeline_run_complete,
    log_pipeline_run_error,
)
from cognee.modules.settings import get_current_settings
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry
from uuid import uuid5, NAMESPACE_OID

from .run_tasks_base import run_tasks_base
from ..tasks.task import Task

logger = get_logger("run_tasks(tasks: [Task], data)")


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, context: dict = None
):
    config = get_current_settings()

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

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

        async for result in run_tasks_base(tasks, data, user, context):
            yield result

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
    user: User = None,
    pipeline_name: str = "unknown_pipeline",
    context: dict = None,
):
    pipeline_id = uuid5(NAMESPACE_OID, pipeline_name)

    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset_id, data)

    yield pipeline_run
    pipeline_run_id = pipeline_run.pipeline_run_id

    try:
        async for _ in run_tasks_with_telemetry(
            tasks=tasks,
            data=data,
            user=user,
            pipeline_name=pipeline_id,
            context=context,
        ):
            pass

        yield await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data
        )

    except Exception as e:
        yield await log_pipeline_run_error(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data, e
        )
        raise e
