from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun


async def get_pipeline_run(pipeline_run_id: UUID):
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = select(PipelineRun).filter(PipelineRun.pipeline_run_id == pipeline_run_id)

        return await session.scalar(query)
