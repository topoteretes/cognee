"""Close pipeline runs that were left STARTED by a process that is gone.

Runs at API startup only. A booting process has no runs of its own in flight,
so a STARTED row it finds cannot be one it is executing. Rows younger than
``COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS`` are still left alone, in case a
sibling process (a rolling deploy) is running them.
"""

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.exceptions import AbandonedPipelineRunError
from cognee.modules.pipelines.methods import get_latest_pipeline_runs_for_all_pipelines
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.log_pipeline_run_error import log_pipeline_run_error
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify.recovery")

STALE_RUN_MIN_AGE_SECONDS = int(os.getenv("COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS", "3600"))

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
    """Roll back and close every pipeline run left STARTED, during API startup.

    For each (dataset, pipeline) pair only the newest row is considered, and
    only when it is ``DATASET_PROCESSING_STARTED``. A run whose newest row is
    already ``ERRORED`` or ``COMPLETED`` stays exactly as it is: an ERRORED run
    was rolled back inline when it failed (see ``run_tasks``), so touching it
    again would repeat the rollback on every restart.

    A candidate is first rolled back with the pipeline's own handler from
    ``ROLLBACK_HANDLERS``, if it has one, then closed with a
    ``DATASET_PROCESSING_ERRORED`` row whose error is ``AbandonedPipelineRunError``.
    The ERRORED row carries the STARTED row's user, tenant, start time, input
    summary, origin and parent operation, so it describes the run that died,
    not the process closing it. The run gate then no longer reports the dataset
    as "already being processed", and the activity feed shows the run as
    abandoned rather than making it disappear. If the rollback fails the run is
    left at STARTED so the next startup retries it.
    """
    db_engine = get_relational_engine()

    try:
        latest_runs = await get_latest_pipeline_runs_for_all_pipelines()
        recovery_candidates = [
            run for run in latest_runs if run.status == PipelineRunStatus.DATASET_PROCESSING_STARTED
        ]
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

        async with db_engine.get_async_session() as session:
            dataset = await session.get(Dataset, pipeline_run.dataset_id)
        if dataset is None:
            logger.warning(
                "Skipping startup recovery for %s run %s: dataset %s not found.",
                pipeline_run.pipeline_name,
                pipeline_run.pipeline_run_id,
                pipeline_run.dataset_id,
            )
            continue

        rollback_handler = ROLLBACK_HANDLERS.get(pipeline_run.pipeline_name)
        try:
            async with set_database_global_context_variables(dataset.id, dataset.owner_id):
                if rollback_handler is not None:
                    await rollback_handler(
                        pipeline_run_id=pipeline_run.pipeline_run_id,
                        dataset=dataset,
                    )
                await _close_as_abandoned(pipeline_run, dataset)
            logger.info(
                "Startup recovery closed %s run %s as abandoned (dataset=%s, rolled_back=%s).",
                pipeline_run.pipeline_name,
                pipeline_run.pipeline_run_id,
                pipeline_run.dataset_id,
                rollback_handler is not None,
            )
        except Exception:
            logger.exception(
                "Startup recovery failed for %s run %s",
                pipeline_run.pipeline_name,
                pipeline_run.pipeline_run_id,
            )


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
