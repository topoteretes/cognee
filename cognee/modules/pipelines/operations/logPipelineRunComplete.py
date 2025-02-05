from uuid import UUID, uuid4
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus


async def logPipelineRunComplete(pipeline_id: UUID, dataset_id: UUID, data: list[Data]):
    if not isinstance(data, (str, list)):
        raise TypeError("data must be either a string or a list of Data objects")

    if isinstance(data, list):
        if all(isinstance(item, Data) for item in data):
            data_info = [str(item.id) for item in data]
        elif all(isinstance(item, str) for item in data):
            data_info = [item for item in data]
        else:
            raise TypeError("All items in the data list must be of type Data or str")
    if isinstance(data, str):
        data_info = data

    pipeline_run_id = uuid4()

    pipeline_run = PipelineRun(
        id=pipeline_run_id,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        run_info={
            "dataset_id": str(dataset_id),
            "data": data_info,
        },
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
