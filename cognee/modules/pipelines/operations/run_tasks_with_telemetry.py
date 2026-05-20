import json
from typing import Optional

from cognee.modules.settings import get_current_settings
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version
from cognee.modules.pipelines.models import PipelineContext

from .run_tasks_single import run_tasks_single
from ..tasks.task import Task


logger = get_logger("run_tasks_with_telemetry()")


def _safe_send_telemetry(event_name: str, user_id, additional_properties: dict) -> None:
    """Emit telemetry without letting an outage mask the original pipeline outcome."""
    try:
        send_telemetry(event_name, user_id, additional_properties=additional_properties)
    except Exception:
        logger.warning("Telemetry emission failed for %s", event_name, exc_info=True)


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, ctx: Optional[PipelineContext] = None
):
    config = get_current_settings()

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        _safe_send_telemetry(
            "Pipeline Run Started",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )

        async for result in run_tasks_single(tasks, data, user, ctx):
            yield result

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        _safe_send_telemetry(
            "Pipeline Run Completed",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )
    except Exception as error:
        logger.error(
            "Pipeline run errored: `%s`\n%s\n",
            pipeline_name,
            str(error),
            exc_info=True,
        )
        _safe_send_telemetry(
            "Pipeline Run Errored",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )

        raise error
