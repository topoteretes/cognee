import json
from typing import Optional

from cognee.context_global_variables import current_pipeline_run_id
from cognee.modules.settings import get_current_settings
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import as_uuid, send_telemetry
from cognee import __version__ as cognee_version
from cognee.modules.pipelines.models import PipelineContext

from .run_tasks_base import run_tasks_base
from ..tasks.task import Task


logger = get_logger("run_tasks_with_telemetry()")


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, ctx: Optional[PipelineContext] = None
):
    config = get_current_settings()

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

    # Publish the run id for the duration of the pipeline so every telemetry
    # event raised inside it — here and in nested tasks — can be correlated with
    # the PipelineRun row, without threading the id through each emitter.
    run_id_token = None
    if ctx is not None and getattr(ctx, "pipeline_run_id", None):
        run_id_token = current_pipeline_run_id.set(as_uuid(ctx.pipeline_run_id))

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

        async for result in run_tasks_base(tasks, data, user, ctx):
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
    except Exception as error:
        logger.error(
            "Pipeline run errored: `%s`\n%s\n",
            pipeline_name,
            str(error),
            exc_info=True,
        )
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

        raise error
    finally:
        if run_id_token is not None:
            current_pipeline_run_id.reset(run_id_token)
