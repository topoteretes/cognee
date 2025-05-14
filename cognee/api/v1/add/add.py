from typing import Union, BinaryIO, List, Optional
from uuid import NAMESPACE_OID, uuid5
from cognee.modules.pipelines.operations.log_pipeline_run_initiated import log_pipeline_run_initiated
from cognee.modules.users.models import User
from cognee.modules.pipelines import Task
from cognee.tasks.ingestion import ingest_data, resolve_data_directories
from cognee.modules.pipelines import cognee_pipeline


async def add(
    data: Union[BinaryIO, list[BinaryIO], str, list[str]],
    dataset_name: str = "main_dataset",
    user: User = None,
    node_set: Optional[List[str]] = None,
):
    tasks = [Task(resolve_data_directories), Task(ingest_data, dataset_name, user, node_set)]

    # Refresh the add pipeline status so data is added to a dataset
    dataset_id = uuid5(NAMESPACE_OID, f"{dataset_name}{str(user.id)}")
    await log_pipeline_run_initiated(
        pipeline_id=uuid5(NAMESPACE_OID, "add_pipeline"),
        pipeline_name="add_pipeline",
        dataset_id=dataset_id,
    )

    await cognee_pipeline(
        tasks=tasks, datasets=dataset_name, data=data, user=user, pipeline_name="add_pipeline"
    )

    # Refresh the cognify pipeline status so UI shows the correct status
    dataset_id = uuid5(NAMESPACE_OID, f"{dataset_name}{str(user.id)}")
    await log_pipeline_run_initiated(
        pipeline_id=uuid5(NAMESPACE_OID, "cognify_pipeline"),
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
    )
