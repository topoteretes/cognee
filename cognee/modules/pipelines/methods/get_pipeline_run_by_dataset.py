from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models import PipelineRun, PipelineRunStatus


def _latest_run_per_dataset_query(dataset_ids: list[UUID] | None, pipeline_name: str):
    """The newest PipelineRun row per dataset, ranked by created_at desc.

    dataset_ids=None means every dataset, not none of them.
    """
    query = select(
        PipelineRun,
        func.row_number()
        .over(
            partition_by=PipelineRun.dataset_id,
            order_by=PipelineRun.created_at.desc(),
        )
        .label("rn"),
    ).filter(PipelineRun.pipeline_name == pipeline_name)
    if dataset_ids is not None:
        query = query.filter(PipelineRun.dataset_id.in_(dataset_ids))
    ranked_runs = query.subquery()
    aliased_run = aliased(PipelineRun, ranked_runs)
    return select(aliased_run).filter(ranked_runs.c.rn == 1)


async def get_pipeline_run_by_dataset(dataset_id: UUID, pipeline_name: str):
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = _latest_run_per_dataset_query([dataset_id], pipeline_name)
        run = (await session.execute(query)).scalars().first()

        return run


async def get_latest_pipeline_runs_by_datasets(
    dataset_ids: list[UUID] | None, pipeline_name: str
) -> dict[UUID, PipelineRun]:
    """The batched sibling of get_pipeline_run_by_dataset: the newest run per
    dataset, in one query, keyed by dataset_id.

    dataset_ids=None returns the latest run for every dataset that has one,
    not an empty result — pass a list (possibly empty) to scope it.
    """
    if dataset_ids is not None and not dataset_ids:
        return {}

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = _latest_run_per_dataset_query(dataset_ids, pipeline_name)
        runs = (await session.execute(query)).scalars().all()

    return {run.dataset_id: run for run in runs}


async def get_unterminated_pipeline_runs() -> list[PipelineRun]:
    """Every run whose newest row is still STARTED: the runs no terminal row closed.

    Keyed by pipeline_run_id, not by dataset: a run abandoned while a newer run of
    the same pipeline on the same dataset later completed must still be found,
    and "newest row per dataset" would hide it behind that newer run forever.
    Rows without a pipeline_name (operation records) are never included. The
    returned row is the run's newest one, so ``created_at`` is its last activity.
    """
    ranked = (
        select(
            PipelineRun,
            func.row_number()
            .over(
                partition_by=PipelineRun.pipeline_run_id,
                order_by=PipelineRun.created_at.desc(),
            )
            .label("rn"),
        )
        .filter(PipelineRun.pipeline_name.isnot(None))
        .subquery()
    )
    newest_per_run = aliased(PipelineRun, ranked)
    query = select(newest_per_run).filter(
        ranked.c.rn == 1,
        ranked.c.status == PipelineRunStatus.DATASET_PROCESSING_STARTED,
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        return list((await session.execute(query)).scalars().all())
