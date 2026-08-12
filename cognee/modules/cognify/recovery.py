import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.methods import reset_pipeline_run_status
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.utils import is_owner_process_alive
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify.recovery")

# A cognify run is only treated as "stale" (abandoned by a crashed process) once
# it has gone this long without reporting progress. This guards against rolling
# back a run that is still actively executing in another live worker/replica
# (e.g. during a rolling deploy or a multi-process deployment sharing one
# database). Override via env when a single cognify task legitimately takes
# longer than the default to complete.
STALE_RUN_MIN_AGE_SECONDS = int(os.getenv("COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS", "3600"))


def _last_progress_at(pipeline_run):
    """Return the most recent evidence that this run was alive.

    ``last_heartbeat_at`` is stamped by the pipeline as it completes each task
    (see ``operations.heartbeat_pipeline_run``), so it measures progress rather
    than elapsed time. Runs that predate the heartbeat column -- and runs that
    died before completing their first task -- have none, and fall back to
    ``created_at``, which is the original age-only behaviour.
    """
    return getattr(pipeline_run, "last_heartbeat_at", None) or getattr(
        pipeline_run, "created_at", None
    )


def _has_stalled_past_threshold(last_progress_at) -> bool:
    """Return True if the run has been silent long enough to be considered stale.

    Age is deliberately not the signal here. With a local LLM a cognify run can
    legitimately take days, so "started long ago" says nothing about whether the
    run is alive; "has not finished a task in an hour" does. A run that keeps
    completing tasks keeps pushing its own cutoff forward and is never a
    recovery candidate, however long it runs in total.

    When there is no timestamp at all (e.g. legacy rows) we cannot prove the run
    is alive, so we conservatively allow recovery to proceed.
    """
    if last_progress_at is None:
        return True

    if last_progress_at.tzinfo is None:
        last_progress_at = last_progress_at.replace(tzinfo=timezone.utc)

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=STALE_RUN_MIN_AGE_SECONDS)
    return last_progress_at <= cutoff


async def recover_stale_cognify_runs_on_startup() -> None:
    """Recover latest non-terminal cognify runs during API startup.

    Startup recovery is intentionally limited to API lifespan initialization,
    before any new pipeline processing starts.

    Only runs whose latest status is ``DATASET_PROCESSING_STARTED`` are
    recovered: an ``ERRORED`` run has already been rolled back inline at error
    time (see ``run_tasks``), so re-selecting it here would repeat the rollback
    on every restart. After a successful rollback the dataset's pipeline status
    is reset to ``DATASET_PROCESSING_INITIATED`` so it is no longer reported as
    "already being processed" and can be cognified again.

    Liveness is decided in two steps. If the run recorded an owner on a node
    this process can see, the operating system answers exactly: a live pid
    means the run is live no matter how quiet it has been, and a dead pid means
    the run cannot resume no matter how recently it reported progress. Only
    when ownership is unknown or belongs to another node does the decision fall
    back to the progress heartbeat and its threshold, so a run that is still
    working is never rolled back out from under a live worker.
    """
    db_engine = get_relational_engine()

    try:
        async with db_engine.get_async_session() as session:
            latest_per_dataset = (
                select(
                    PipelineRun,
                    func.row_number()
                    .over(
                        partition_by=PipelineRun.dataset_id,
                        order_by=PipelineRun.created_at.desc(),
                    )
                    .label("rn"),
                )
                .where(PipelineRun.pipeline_name == "cognify_pipeline")
                .subquery()
            )

            latest_run = aliased(PipelineRun, latest_per_dataset)
            recovery_candidates = (
                (
                    await session.execute(
                        select(latest_run)
                        .where(latest_per_dataset.c.rn == 1)
                        .where(latest_run.status == PipelineRunStatus.DATASET_PROCESSING_STARTED)
                    )
                )
                .scalars()
                .all()
            )
    except Exception:
        logger.error("Failed to recover latest cognify run which did not successfully finish.")
        return

    for pipeline_run in recovery_candidates:
        owner_alive = is_owner_process_alive(
            getattr(pipeline_run, "owner_node_id", None),
            getattr(pipeline_run, "owner_pid", None),
        )

        if owner_alive is True:
            logger.info(
                "Skipping startup recovery for run %s: its owning process (node=%s pid=%s) "
                "is still running, so the run is live regardless of heartbeat age.",
                pipeline_run.pipeline_run_id,
                pipeline_run.owner_node_id,
                pipeline_run.owner_pid,
            )
            continue

        if owner_alive is False:
            # The owner died on this node. No threshold applies: a run whose
            # process is gone cannot resume, however recently it reported
            # progress. Recovering now also releases the dataset immediately
            # rather than leaving it blocked as "already being processed".
            logger.info(
                "Recovering run %s: its owning process (node=%s pid=%s) is gone.",
                pipeline_run.pipeline_run_id,
                pipeline_run.owner_node_id,
                pipeline_run.owner_pid,
            )
        else:
            # Ownership is unknown or belongs to another node, so fall back to
            # the progress heartbeat.
            last_progress_at = _last_progress_at(pipeline_run)

            if not _has_stalled_past_threshold(last_progress_at):
                logger.info(
                    "Skipping startup recovery for run %s: last reported progress at %s "
                    "(less than %ds ago), treating it as a live run rather than a stale one.",
                    pipeline_run.pipeline_run_id,
                    last_progress_at,
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
                # Clear the lingering STARTED status so a re-run is not blocked by
                # check_pipeline_run_qualification ("already being processed").
                await reset_pipeline_run_status(
                    user_id=dataset.owner_id,
                    dataset_id=dataset.id,
                    pipeline_name="cognify_pipeline",
                )
            logger.info(
                "Startup recovery completed for cognify run %s (dataset=%s).",
                pipeline_run.pipeline_run_id,
                pipeline_run.dataset_id,
            )
        except Exception as error:
            logger.error(
                "Startup recovery failed for cognify run %s: %s",
                pipeline_run.pipeline_run_id,
                error,
                exc_info=True,
            )
