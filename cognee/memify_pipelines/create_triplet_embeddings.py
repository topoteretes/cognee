from typing import Any

from cognee import memify
from cognee.context_global_variables import (
    set_database_global_context_variables,
)
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.shared.logging_utils import get_logger
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.tasks.memify.get_triplet_datapoints import get_triplet_datapoints
from cognee.tasks.storage import index_data_points

logger = get_logger("create_triplet_embeddings")


async def create_triplet_embeddings(
    user: User,
    dataset: str = "main_dataset",
    run_in_background: bool = False,
    triplets_batch_size: int = 100,
) -> dict[str, Any]:
    dataset_to_write = await get_authorized_existing_datasets(
        user=user, datasets=[dataset], permission_type="write"
    )

    if not dataset_to_write:
        raise CogneeValidationError(
            message=f"User does not have write access to dataset: {dataset}",
            log=False,
        )

    await set_database_global_context_variables(
        dataset_to_write[0].id, dataset_to_write[0].owner_id
    )

    extraction_tasks = [Task(get_triplet_datapoints, triplets_batch_size=triplets_batch_size)]

    enrichment_tasks = [
        Task(index_data_points, task_config={"batch_size": triplets_batch_size}),
    ]

    result = await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset_to_write[0].id,
        data=[{}],
        user=user,
        run_in_background=run_in_background,
    )

    return result
