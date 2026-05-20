from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify.recovery")


async def recover_stale_cognify_runs_on_startup() -> None:
    """Recover latest non-terminal cognify runs during API startup.

    Startup recovery is intentionally limited to API lifespan initialization,
    before any new pipeline processing starts.
    """
    db_engine = get_relational_engine()

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
                    .where(
                        latest_run.status.in_(
                            [
                                PipelineRunStatus.DATASET_PROCESSING_STARTED,
                                PipelineRunStatus.DATASET_PROCESSING_ERRORED,
                            ]
                        )
                    )
                )
            )
            .scalars()
            .all()
        )

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
