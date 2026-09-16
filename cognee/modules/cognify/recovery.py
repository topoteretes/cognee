import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.exceptions import AbandonedPipelineRunError
from cognee.modules.pipelines.methods import get_latest_pipeline_runs_by_datasets
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.log_pipeline_run_error import log_pipeline_run_error
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify.recovery")

# A cognify run is only treated as "stale" (abandoned by a crashed process) once
# it has stayed in a non-terminal state longer than this threshold. This guards
# against rolling back a run that is still actively executing in another live
# worker/replica (e.g. during a rolling deploy or a multi-process deployment
# sharing one database). A heartbeat/lease would be more precise; an age
# threshold is a pragmatic guard. Override via env when long-running cognify
# jobs legitimately exceed the default.
STALE_RUN_MIN_AGE_SECONDS = int(os.getenv("COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS", "3600"))


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


async def recover_stale_cognify_runs_on_startup() -> None:
    """Recover latest non-terminal cognify runs during API startup.

    Startup recovery is intentionally limited to API lifespan initialization,
    before any new pipeline processing starts.

    Only runs whose latest status is ``DATASET_PROCESSING_STARTED`` are
    recovered: an ``ERRORED`` run has already been rolled back inline at error
    time (see ``run_tasks``), so re-selecting it here would repeat the rollback
    on every restart. After a successful rollback the run is closed with a
    ``DATASET_PROCESSING_ERRORED`` row whose error is ``AbandonedPipelineRunError``:
    the run gate no longer reports the dataset as "already being processed", so it
    can be cognified again, and the activity feed shows the run as abandoned rather
    than making it disappear. If the rollback fails the run is left at STARTED so
    the next startup retries it.
    """
    db_engine = get_relational_engine()

    try:
        latest_per_dataset = await get_latest_pipeline_runs_by_datasets(None, "cognify_pipeline")
        recovery_candidates = [
            run
            for run in latest_per_dataset.values()
            if run.status == PipelineRunStatus.DATASET_PROCESSING_STARTED
        ]
    except Exception:
        logger.exception("Failed to recover latest cognify run which did not successfully finish.")
        return

    for pipeline_run in recovery_candidates:
        if not _is_older_than_threshold(getattr(pipeline_run, "created_at", None)):
            logger.info(
                "Skipping startup recovery for run %s: started less than %ds ago, "
                "treating it as a live run rather than a stale one.",
                pipeline_run.pipeline_run_id,
                STALE_RUN_MIN_AGE_SECONDS,
            )
            continue

        async with db_engine.get_async_session() as session:
            dataset = await session.get(Dataset, pipeline_run.dataset_id)

        if dataset is None:
            logger.warning(
                "Skipping startup recovery for run %s: dataset %s not found.",
                pipeline_run.pipeline_run_id,
                pipeline_run.dataset_id,
            )
            continue

        try:
            async with set_database_global_context_variables(dataset.id, dataset.owner_id):
                await cognify_rollback_handler(
                    pipeline_run_id=pipeline_run.pipeline_run_id,
                    dataset=dataset,
                )
                # Close the run. The newest row for the dataset is now ERRORED, so
                # check_pipeline_run_qualification lets a re-run through, and the
                # error class records that the run was abandoned, not failed.
                await log_pipeline_run_error(
                    pipeline_run_id=pipeline_run.pipeline_run_id,
                    pipeline_id=pipeline_run.pipeline_id,
                    pipeline_name="cognify_pipeline",
                    dataset_id=dataset.id,
                    data=None,
                    e=AbandonedPipelineRunError(),
                    user=SimpleNamespace(
                        id=pipeline_run.user_id,
                        tenant_id=getattr(pipeline_run, "tenant_id", None),
                    )
                    if getattr(pipeline_run, "user_id", None)
                    else None,
                    started_at=getattr(pipeline_run, "started_at", None),
                )
            logger.info(
                "Startup recovery completed for cognify run %s (dataset=%s).",
                pipeline_run.pipeline_run_id,
                pipeline_run.dataset_id,
            )
        except Exception:
            logger.exception(
                "Startup recovery failed for cognify run %s",
                pipeline_run.pipeline_run_id,
            )
