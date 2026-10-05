"""Recover old runs only after proving their writer no longer owns them.

The API checks at startup and periodically. The age threshold is an eligibility
floor; a local dataset lock and the run's OS ownership marker establish whether
recovery may proceed. Unverifiable legacy/remote rows are left untouched.
"""

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.locks import get_dataset_lock
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.exceptions import AbandonedPipelineRunError
from cognee.modules.pipelines.methods import get_unterminated_pipeline_runs
from cognee.modules.pipelines.methods.get_pipeline_run import get_latest_pipeline_run
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.log_pipeline_run_error import log_pipeline_run_error
from cognee.modules.pipelines.operations.run_tasks_with_telemetry import (
    PIPELINE_RUN_ERRORED,
    pipeline_run_telemetry_properties,
)
from cognee.modules.pipelines.run_ownership import claim_run_ownership
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry, telemetry_guard

logger = get_logger("cognify.recovery")

STALE_RUN_MIN_AGE_SECONDS = int(os.getenv("COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS", "3600"))
if STALE_RUN_MIN_AGE_SECONDS < 0:
    raise ValueError("COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS must be nonnegative")
RECOVERY_POLL_SECONDS = 60

# The rollback each pipeline supplies for its own failed runs (the same policy
# run_tasks applies when a run errors inline). A pipeline without an entry has
# nothing to roll back at error time either, so at startup it is only closed.
ROLLBACK_HANDLERS = {
    "cognify_pipeline": cognify_rollback_handler,
}


def _is_older_than_threshold(created_at) -> bool:
    """Return True if the run started long enough ago to be considered stale.

    When ``created_at`` is missing (e.g. legacy rows) we cannot prove the run is
    young, so we conservatively allow recovery to proceed.
    """
    if created_at is None:
        return True
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=STALE_RUN_MIN_AGE_SECONDS)
    return created_at <= cutoff


async def recover_stale_pipeline_runs_on_startup() -> None:
    """Roll back and close eligible abandoned runs.

    Age is only an eligibility floor. Recovery must claim the run's existing
    OS ownership lock and re-read its status before rollback. Legacy rows and
    workers whose marker is not on this filesystem cannot be proven abandoned
    and are left alone. The API lifespan revisits deferred runs periodically.
    """
    try:
        recovery_candidates = await get_unterminated_pipeline_runs()
    except Exception:
        logger.exception("Failed to load pipeline runs for startup recovery.")
        return

    for pipeline_run in recovery_candidates:
        if not _is_older_than_threshold(getattr(pipeline_run, "created_at", None)):
            logger.info(
                "Skipping startup recovery for %s run %s: started less than %ds ago, "
                "treating it as a live run rather than a stale one.",
                pipeline_run.pipeline_name,
                pipeline_run.pipeline_run_id,
                STALE_RUN_MIN_AGE_SECONDS,
            )
            continue

        try:
            lock = await get_dataset_lock(pipeline_run.dataset_id)
            if lock.locked():
                continue
            async with lock:
                with claim_run_ownership(pipeline_run) as ownership:
                    if ownership is None:
                        logger.debug(
                            "Run %s is active or its ownership cannot be verified; skipping recovery",
                            pipeline_run.pipeline_run_id,
                        )
                        continue
                    # A writer/recoverer may have finished since candidate selection.
                    current = await get_latest_pipeline_run(pipeline_run.pipeline_run_id)
                    if (
                        current is None
                        or current.status != PipelineRunStatus.DATASET_PROCESSING_STARTED
                    ):
                        ownership.closed = True
                        continue
                    async with get_relational_engine().get_async_session() as session:
                        dataset = await session.get(Dataset, current.dataset_id)
                    if dataset is None:
                        logger.warning("Recovery dataset %s no longer exists", current.dataset_id)
                        continue
                    rollback_handler = ROLLBACK_HANDLERS.get(current.pipeline_name)
                    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
                        if rollback_handler is not None:
                            await rollback_handler(
                                pipeline_run_id=current.pipeline_run_id,
                                dataset=dataset,
                                keep_completed_data=True,
                            )
                        await _close_as_abandoned(current, dataset)
                    ownership.closed = True
                    logger.info("Recovery closed run %s as abandoned", current.pipeline_run_id)
                    _send_abandoned_run_telemetry(current)
        except Exception:
            logger.exception("Recovery failed for run %s; will retry", pipeline_run.pipeline_run_id)


async def _recheck_stale_runs() -> None:
    while True:
        await asyncio.sleep(RECOVERY_POLL_SECONDS)
        await recover_stale_pipeline_runs_on_startup()


@asynccontextmanager
async def pipeline_recovery_service():
    """Run recovery now and revisit candidates until API shutdown."""
    await recover_stale_pipeline_runs_on_startup()
    task = asyncio.create_task(_recheck_stale_runs())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def _close_as_abandoned(pipeline_run, dataset) -> None:
    """Write the ERRORED row for ``pipeline_run``, carrying the STARTED row's metadata."""
    user_id = getattr(pipeline_run, "user_id", None)
    await log_pipeline_run_error(
        pipeline_run_id=pipeline_run.pipeline_run_id,
        pipeline_id=pipeline_run.pipeline_id,
        pipeline_name=pipeline_run.pipeline_name,
        dataset_id=dataset.id,
        data=None,
        e=AbandonedPipelineRunError(),
        user=SimpleNamespace(id=user_id, tenant_id=getattr(pipeline_run, "tenant_id", None))
        if user_id
        else None,
        started_at=getattr(pipeline_run, "started_at", None),
        data_info=(getattr(pipeline_run, "run_info", None) or {}).get("data"),
        origin=getattr(pipeline_run, "origin", None),
        parent_operation_id=getattr(pipeline_run, "parent_operation_id", None),
    )


def _send_abandoned_run_telemetry(pipeline_run) -> None:
    """Emit the terminal telemetry event the dead process never sent.

    The run's ``Pipeline Run Started`` went out when it began; without this the
    warehouse counts an abandoned run as a silent gap forever while the local
    ``pipeline_runs`` table shows it closed. ``pipeline_name`` is the pipeline
    id, as the live emitter sends it; ``exception_type`` is the class the ERRORED
    row carries. Never raises: telemetry must not turn a successful recovery
    into a logged failure.
    """
    with telemetry_guard():
        user_id = getattr(pipeline_run, "user_id", None)
        tenant_id = getattr(pipeline_run, "tenant_id", None)
        properties = pipeline_run_telemetry_properties(
            pipeline_run.pipeline_id, pipeline_run.pipeline_run_id, tenant_id, recovered=True
        ) | {
            "exception_type": AbandonedPipelineRunError.__name__,
            "recovered_at_startup": True,
            "pipeline_event_scope": "run",
        }
        send_telemetry(
            PIPELINE_RUN_ERRORED,
            SimpleNamespace(id=user_id, tenant_id=tenant_id) if user_id else None,
            additional_properties=properties,
        )
