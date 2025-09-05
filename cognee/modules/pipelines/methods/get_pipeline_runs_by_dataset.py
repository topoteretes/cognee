from uuid import UUID
from typing import Optional
from sqlalchemy import select, func
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import PipelineRun


async def get_pipeline_runs_by_dataset(dataset_id: UUID):
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = (
            select(
                PipelineRun,
                func.row_number()
                .over(
                    partition_by=(PipelineRun.dataset_id, PipelineRun.pipeline_name),
                    order_by=PipelineRun.created_at.desc(),
                )
                .label("rn"),
            )
            .filter(PipelineRun.dataset_id == dataset_id)
            .subquery()
        )

        aliased_pipeline_run = aliased(PipelineRun, query)

        latest_run = select(aliased_pipeline_run).filter(query.c.rn == 1)

        runs = (await session.execute(latest_run)).scalars().all()

        return runs
