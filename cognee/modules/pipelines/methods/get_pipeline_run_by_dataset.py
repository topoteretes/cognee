from typing import Dict, List, Optional
from uuid import UUID
from sqlalchemy import select, func
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import PipelineRun
from sqlalchemy.orm import aliased


def _latest_run_per_dataset_query(dataset_ids: Optional[List[UUID]], pipeline_name: str):
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
    dataset_ids: Optional[List[UUID]], pipeline_name: str
) -> Dict[UUID, PipelineRun]:
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
