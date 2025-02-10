from uuid import UUID
from sqlalchemy import func, select
from sqlalchemy.orm import aliased
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import PipelineRun


async def get_pipeline_status(pipeline_ids: list[UUID]):
    db_engine = get_relational_engine()
    dialect = db_engine.engine.dialect.name

    async with db_engine.get_async_session() as session:
        if dialect == "sqlite":
            dataset_id_column = func.json_extract(PipelineRun.run_info, "$.dataset_id")
        else:
            dataset_id_column = PipelineRun.run_info.op("->>")("dataset_id")

        query = (
            select(
                PipelineRun,
                func.row_number()
                .over(
                    partition_by=dataset_id_column,
                    order_by=PipelineRun.created_at.desc(),
                )
                .label("rn"),
            )
            .filter(dataset_id_column.in_([str(id) for id in pipeline_ids]))
            .subquery()
        )

        aliased_pipeline_run = aliased(PipelineRun, query)
        latest_runs = select(aliased_pipeline_run).filter(query.c.rn == 1)

        runs = (await session.execute(latest_runs)).scalars().all()

        pipeline_statuses = {run.run_info["dataset_id"]: run.status for run in runs}

        return pipeline_statuses
