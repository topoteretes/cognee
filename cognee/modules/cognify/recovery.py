from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.exceptions import AbandonedPipelineRunError
from cognee.modules.pipelines.methods import get_latest_pipeline_runs_by_datasets
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations import log_pipeline_run_error
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify.recovery")


async def recover_stale_cognify_runs_on_startup() -> None:
    """Close cognify runs whose process did not survive, during API startup.

    A pipeline executes inside the API process. So a STARTED row with no
    terminal row, found while that process is coming back up, belonged to a
    process that is gone: the restart is the evidence. Nothing about the run's
    age is consulted, which is the point. A run on a local model can
    legitimately take days, and an age threshold would either roll that run
    back or, set high enough not to, leave a genuinely dead run reported as
    processing until some later boot.

    That reasoning assumes one instance per relational database, which is what
    on-demand cognee servers are. If several processes ever share a database,
    a restarting one cannot tell its own dead run from another's live one, and
    this needs a liveness signal on the row rather than the restart.

    Only runs whose latest status is ``DATASET_PROCESSING_STARTED`` are
    recovered: an ``ERRORED`` run has already been rolled back inline at error
    time (see ``run_tasks``), so re-selecting it here would repeat the rollback
    on every restart. After the rollback the run is closed as
    ``DATASET_PROCESSING_ERRORED`` carrying ``AbandonedPipelineRunError``, so
    the dataset stops reporting work that is not happening, the run keeps its
    identity, and readers that care can tell killed from failed by the error
    class rather than by a status of their own.
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
                # Record what happened, after the rollback rather than
                # before it: the STARTED row is the retry token, so a rollback
                # that raises leaves the run open for the next boot to finish
                # unwinding instead of marking it closed over a half-deleted
                # graph.
                await log_pipeline_run_error(
                    pipeline_run_id=pipeline_run.pipeline_run_id,
                    pipeline_id=pipeline_run.pipeline_id,
                    pipeline_name="cognify_pipeline",
                    dataset_id=dataset.id,
                    data=None,
                    e=AbandonedPipelineRunError(),
                    started_at=getattr(pipeline_run, "started_at", None),
                    # The STARTED row already holds a summarized payload;
                    # summarizing it again would re-truncate a truncated value.
                    data_info=(pipeline_run.run_info or {}).get("data"),
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
