from typing import Union, BinaryIO, List, Optional

from cognee.modules.pipelines import Task
from cognee.modules.users.models import User
from cognee.modules.pipelines import cognee_pipeline
from cognee.tasks.ingestion import ingest_data, resolve_data_directories


async def add(
    data: Union[BinaryIO, list[BinaryIO], str, list[str]],
    dataset_name: str = "main_dataset",
    user: User = None,
    node_set: Optional[List[str]] = None,
):
    tasks = [Task(resolve_data_directories), Task(ingest_data, dataset_name, user, node_set)]

    await cognee_pipeline(
        tasks=tasks, datasets=dataset_name, data=data, user=user, pipeline_name="add_pipeline"
    )
