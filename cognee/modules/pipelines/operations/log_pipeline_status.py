from cognee.infrastructure.databases.relational import get_relational_engine
from ..models.PipelineRun import PipelineRun

async def log_pipeline_status(run_name: str, status: str, run_info: dict):
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(PipelineRun(
            run_name = run_name,
            status = status,
            run_info = run_info,
        ))

        await session.commit()
