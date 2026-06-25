import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.decay_weights import decay_weights

logger = get_logger("decay_weights_pipeline")


async def decay_weights_pipeline(
    user: User,
    dataset: str = "main_dataset",
    decay_rate: float = 0.95,
    prune_threshold: float = 0.1,
    run_in_background: bool = False,
):
    """
    Decay and pruning pipeline to perform memory maintenance tasks in the background or blocking mode.
    """
    dataset_to_write = await get_authorized_existing_datasets(
        user=user,
        datasets=[dataset],
        permission_type="write",
    )
    if not dataset_to_write:
        raise ValueError(f"User does not have write access to dataset: {dataset}")

    async with set_database_global_context_variables(
        dataset_to_write[0].id, dataset_to_write[0].owner_id
    ):
        extraction_tasks = []
        enrichment_tasks = [
            Task(decay_weights, decay_rate=decay_rate, prune_threshold=prune_threshold)
        ]

        result = await cognee.memify(
            extraction_tasks=extraction_tasks,
            enrichment_tasks=enrichment_tasks,
            dataset=dataset_to_write[0].id,
            data=[{}],
            user=user,
            run_in_background=run_in_background,
        )

    logger.info(
        "Weight decay memify pipeline completed",
        extra={
            "dataset_id": str(dataset_to_write[0].id),
            "decay_rate": decay_rate,
            "prune_threshold": prune_threshold,
        },
    )

    return result
