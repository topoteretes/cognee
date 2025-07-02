from uuid import UUID
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
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    dataset_id: UUID = None,
):
    tasks = [
        Task(resolve_data_directories),
        Task(ingest_data, dataset_name, user, node_set, dataset_id),
    ]

    pipeline_run_info = None

    async for run_info in cognee_pipeline(
        tasks=tasks,
        datasets=dataset_id if dataset_id else dataset_name,
        data=data,
        user=user,
        pipeline_name="add_pipeline",
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
    ):
        pipeline_run_info = run_info

    return pipeline_run_info
