import json

from cognee import __version__ as cognee_version
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.settings import get_current_settings
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry, telemetry_exception_type

from ..tasks.task import Task
from .run_tasks_base import run_tasks_base

logger = get_logger("run_tasks_with_telemetry()")

PIPELINE_RUN_STARTED = "Pipeline Run Started"
PIPELINE_RUN_COMPLETED = "Pipeline Run Completed"
PIPELINE_RUN_ERRORED = "Pipeline Run Errored"


def tenant_label(tenant_id) -> str:
    return str(tenant_id) if tenant_id else "Single User Tenant"


def pipeline_run_telemetry_properties(pipeline_name, pipeline_run_id, tenant_id) -> dict:
    """The properties every ``Pipeline Run *`` event carries.

    ``pipeline_run_id`` is the run's random UUID (``generate_pipeline_run_id``):
    the join key between a run's Started event and its terminal one, without
    which the warehouse can only compare counts. Startup recovery
    (``cognee.modules.cognify.recovery``) builds the Errored event for an
    abandoned run from this same function so the two emitters cannot drift.
    """
    properties = {
        "pipeline_name": str(pipeline_name),
        "cognee_version": cognee_version,
        "tenant_id": tenant_label(tenant_id),
    }
    if pipeline_run_id is not None:
        properties["pipeline_run_id"] = str(pipeline_run_id)
    return properties | get_current_settings()


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, ctx: PipelineContext | None = None
):
    properties = pipeline_run_telemetry_properties(
        pipeline_name, ctx.pipeline_run_id if ctx else None, user.tenant_id
    )

    logger.debug(
        "\nRunning pipeline with configuration:\n%s\n",
        json.dumps(properties, indent=1, default=str),
    )

    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        send_telemetry(PIPELINE_RUN_STARTED, user, additional_properties=dict(properties))

        async for result in run_tasks_base(tasks, data, user, ctx):
            yield result

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        send_telemetry(PIPELINE_RUN_COMPLETED, user, additional_properties=dict(properties))
    except BaseException as error:
        # asyncio.CancelledError and GeneratorExit are BaseExceptions, not
        # Exceptions: a run cancelled by a shutdown, or this generator closed by
        # a consumer that stopped iterating, used to leave a Started event with
        # no terminal one — the "silent gap" in the warehouse. Same reasoning as
        # run_tasks's CLO-365 handler. Re-raised below either way, so
        # cancellation still propagates.
        if isinstance(error, Exception):
            logger.exception("Pipeline run errored: `%s`\n", pipeline_name)
        else:
            logger.info("Pipeline run cancelled: `%s` (%s)", pipeline_name, type(error).__name__)
        send_telemetry(
            PIPELINE_RUN_ERRORED,
            user,
            additional_properties=properties | {"exception_type": telemetry_exception_type(error)},
        )

        raise
