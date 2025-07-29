import json

from cognee.modules.settings import get_current_settings
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

from .run_tasks_base import run_tasks_base
from ..tasks.task import Task


logger = get_logger("run_tasks_with_telemetry()")


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, context: dict = None, telemetry_handler=None
):
    config = get_current_settings()

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

    # Use injected telemetry_handler or fallback to send_telemetry
    handler = telemetry_handler if telemetry_handler else send_telemetry

    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        handler(
            "Pipeline Run Started",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
            }
            | config,
        )

        async for result in run_tasks_base(tasks, data, user, context, handler):
            yield result

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        handler(
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
        handler(
            "Pipeline Run Errored",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
            }
            | config,
        )

        raise error
