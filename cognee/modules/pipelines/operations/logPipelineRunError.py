from uuid import UUID, uuid4
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from typing import Any


async def logPipelineRunError(pipeline_id: str, dataset_id: UUID, data: Any, e: Exception):
    if not data:
        data_info = "None"
    elif isinstance(data, list) and all(isinstance(item, Data) for item in data):
        data_info = [str(item.id) for item in data]
    else:
        data_info = str(data)

    pipeline_run_id = uuid4()

    pipeline_run = PipelineRun(
        id=pipeline_run_id,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_ERRORED,
        run_info={
            "dataset_id": str(dataset_id),
            "data": data_info,
            "error": str(e),
        },
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
