import json
from typing import Optional

from cognee.modules.settings import get_current_settings
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version
from cognee.modules.pipelines.models import PipelineContext

from .run_tasks_base import run_tasks_base
from ..tasks.task import Task


logger = get_logger("run_tasks_with_telemetry()")


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, ctx: Optional[PipelineContext] = None
):
    config = get_current_settings()
    run_id = ctx.pipeline_run_id if ctx else "N/A"
    pname = ctx.pipeline_name if ctx else pipeline_name

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

    try:
        logger.info("Processing data item for pipeline `%s` (run: `%s`)", pname, run_id)
        send_telemetry(
            "Processing Data Item",
            user.id,
            additional_properties={
                "pipeline_name": str(pname),
                "pipeline_run_id": str(run_id),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )

        async for result in run_tasks_base(tasks, data, user, ctx):
            yield result

        logger.info("Finished processing data item for pipeline `%s` (run: `%s`)", pname, run_id)
        send_telemetry(
            "Data Item Processed",
            user.id,
            additional_properties={
                "pipeline_name": str(pname),
                "pipeline_run_id": str(run_id),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )
    except Exception as error:
        logger.error(
            "Error processing data item for pipeline `%s` (run: `%s`)\n%s\n",
            pname,
            run_id,
            str(error),
            exc_info=True,
        )
        send_telemetry(
            "Data Item Errored",
            user.id,
            additional_properties={
                "pipeline_name": str(pname),
                "pipeline_run_id": str(run_id),
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            }
            | config,
        )

        raise error
