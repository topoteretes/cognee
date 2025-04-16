from typing import Union, BinaryIO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines import run_tasks, Task
from cognee.tasks.ingestion import ingest_data, resolve_data_directories
from uuid import uuid5, NAMESPACE_OID


async def add(
    data: Union[BinaryIO, list[BinaryIO], str, list[str]],
    dataset_name: str = "main_dataset",
    user: User = None,
):
    tasks = [Task(resolve_data_directories), Task(ingest_data, dataset_name, user)]

    from ..cognify.pipeline import cognee_pipeline

    await cognee_pipeline(
        tasks=tasks, datasets=dataset_name, data=data, user=user, pipeline_name="add_pipeline"
    )
