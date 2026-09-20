import asyncio
import json
from contextlib import aclosing

from cognee import __version__ as cognee_version
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.settings import get_current_settings
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

from ..tasks.task import Task
from .run_tasks_base import run_tasks_base

logger = get_logger("run_tasks_with_telemetry()")


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, ctx: PipelineContext | None = None
):
    config = get_current_settings()

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Started",
            user,
            additional_properties={
                "pipeline_name": str(pipeline_name),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )

        async with aclosing(run_tasks_base(tasks, data, user, ctx)) as results:
            async for result in results:
                yield result

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Completed",
            user,
            additional_properties={
                "pipeline_name": str(pipeline_name),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )
    except (Exception, asyncio.CancelledError, GeneratorExit) as exc:
        # Cancellation and early generator closure inherit from BaseException.
        # Report the interrupted item, then preserve cancellation/close semantics.
        if isinstance(exc, Exception):
            logger.exception(
                "Pipeline run errored: `%s`\n",
                pipeline_name,
            )
        else:
            logger.info("Pipeline run interrupted: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Errored",
            user,
            additional_properties={
                "pipeline_name": str(pipeline_name),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )

        raise
