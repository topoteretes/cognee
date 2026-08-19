from uuid import UUID
from sqlalchemy import select, func
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import PipelineRun
from sqlalchemy.orm import aliased


async def _get_latest_pipeline_runs(dataset_ids: list[UUID], pipeline_name: str):
    """One row per dataset: the latest PipelineRun for this pipeline_name.

    Shared by get_pipeline_status and get_pipeline_progress so both agree on
    what "latest run" means — a single ROW_NUMBER() query, not two.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = (
            select(
                PipelineRun,
                func.row_number()
                .over(
                    partition_by=PipelineRun.dataset_id,
                    order_by=PipelineRun.created_at.desc(),
                )
                .label("rn"),
            )
            .filter(PipelineRun.dataset_id.in_(dataset_ids))
            .filter(PipelineRun.pipeline_name == pipeline_name)
            .subquery()
        )

        aliased_pipeline_run = aliased(PipelineRun, query)

        latest_runs = select(aliased_pipeline_run).filter(query.c.rn == 1)

        return (await session.execute(latest_runs)).scalars().all()


async def get_pipeline_status(dataset_ids: list[UUID], pipeline_name: str):
    runs = await _get_latest_pipeline_runs(dataset_ids, pipeline_name)

    return {str(run.dataset_id): run.status for run in runs}


async def get_pipeline_progress(dataset_ids: list[UUID], pipeline_name: str):
    """Same latest-run lookup as get_pipeline_status, plus the in-flight
    progress snapshot (see log_pipeline_run_progress). A separate function —
    and a separate /status/progress endpoint — rather than a flag on
    get_pipeline_status/get_status, so neither call's response shape ever
    branches at runtime on how it was called.

    run_info["progress"] is written incrementally as items/stages complete
    (see run_tasks.py / run_tasks_data_item.py); it is only present while a
    run is STARTED, so absence just means "no progress ticks yet" rather
    than an error.
    """
    runs = await _get_latest_pipeline_runs(dataset_ids, pipeline_name)

    return {
        str(run.dataset_id): {
            "status": run.status,
            "progress": (run.run_info or {}).get("progress"),
        }
        for run in runs
    }
