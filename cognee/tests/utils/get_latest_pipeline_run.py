from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun


async def get_latest_pipeline_run(**filters) -> PipelineRun:
    """The newest pipeline_runs row matching ``filters`` (column=value)."""
    async with get_relational_engine().get_async_session() as session:
        query = (
            select(PipelineRun)
            .filter_by(**filters)
            .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
            .limit(1)
        )
        run = await session.scalar(query)
    assert run is not None, f"No pipeline_runs row matches {filters}."
    return run
