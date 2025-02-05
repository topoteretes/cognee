from uuid import UUID, uuid4
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus


async def logPipelineRunComplete(pipeline_id: UUID, dataset_id: UUID, data: list[Data]):
    pipeline_run_id = uuid4()

    pipeline_run = PipelineRun(
        id=pipeline_run_id,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        run_info={
            "dataset_id": str(dataset_id),
            "data": [str(data.id) for data in data] if isinstance(data, list) else data,
        },
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
