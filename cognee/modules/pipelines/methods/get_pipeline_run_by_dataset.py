from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models import PipelineRun


def _latest_run_per_dataset_query(dataset_ids: list[UUID] | None, pipeline_name: str | None):
    """The newest PipelineRun row per dataset, ranked by created_at desc.

    dataset_ids=None means every dataset, not none of them. pipeline_name=None
    means every pipeline, one newest row per (dataset, pipeline) pair; rows with
    no pipeline_name (operation records) are never included.
    """
    if pipeline_name is None:
        partition_by = (PipelineRun.dataset_id, PipelineRun.pipeline_name)
        name_filter = PipelineRun.pipeline_name.isnot(None)
    else:
        partition_by = PipelineRun.dataset_id
        name_filter = PipelineRun.pipeline_name == pipeline_name
    query = select(
        PipelineRun,
        func.row_number()
        .over(
            partition_by=partition_by,
            order_by=PipelineRun.created_at.desc(),
        )
        .label("rn"),
    ).filter(name_filter)
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


async def get_latest_pipeline_runs_for_all_pipelines() -> list[PipelineRun]:
    """The newest run of every (dataset, pipeline) pair, in one query.

    Startup recovery's view: one row per pair, whichever pipeline it belongs to,
    so a run left STARTED by any pipeline is found, not only cognify's.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = _latest_run_per_dataset_query(None, None)
        return list((await session.execute(query)).scalars().all())
