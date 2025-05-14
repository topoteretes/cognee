from uuid import UUID, uuid4
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus


async def log_pipeline_run_initiated(pipeline_id: str, pipeline_name: str, dataset_id: UUID):
    pipeline_run = PipelineRun(
        pipeline_run_id=uuid4(),
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_INITIATED,
        dataset_id=dataset_id,
        run_info={},
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
