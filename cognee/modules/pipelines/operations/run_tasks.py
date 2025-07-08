import json
from typing import Any
from uuid import UUID, uuid4

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
    PipelineRunYield,
)

from cognee.modules.pipelines.operations import (
    log_pipeline_run_start,
    log_pipeline_run_complete,
    log_pipeline_run_error,
)
from cognee.modules.settings import get_current_settings
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry

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
    dataset_id: UUID,
    data: Any = None,
    user: User = None,
    pipeline_name: str = "unknown_pipeline",
    context: dict = None,
):
    if not user:
        user = get_default_user()

    # Get Dataset object
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        from cognee.modules.data.models import Dataset

        dataset = await session.get(Dataset, dataset_id)

    pipeline_id = generate_pipeline_id(user.id, dataset.id, pipeline_name)

    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset_id, data)

    pipeline_run_id = pipeline_run.pipeline_run_id

    yield PipelineRunStarted(
        pipeline_run_id=pipeline_run_id,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        payload=data,
    )

    try:
        async for result in run_tasks_with_telemetry(
            tasks=tasks,
            data=data,
            user=user,
            pipeline_name=pipeline_id,
            context=context,
        ):
            yield PipelineRunYield(
                pipeline_run_id=pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                payload=result,
            )

        await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data
        )

        yield PipelineRunCompleted(
            pipeline_run_id=pipeline_run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
        )

    except Exception as error:
        await log_pipeline_run_error(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data, error
        )

        yield PipelineRunErrored(
            pipeline_run_id=pipeline_run_id,
            payload=error,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
        )

        raise error
