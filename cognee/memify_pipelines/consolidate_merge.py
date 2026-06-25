import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.consolidate_merge import consolidate_merge

logger = get_logger("consolidate_merge_pipeline")


async def consolidate_merge_pipeline(
    user: User,
    dataset: str = "main_dataset",
    similarity_threshold: float = 0.85,
    run_in_background: bool = False,
):
    """
    Pipeline wrapper for Consolidate & Merge near-duplicate entities memory task.
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
        enrichment_tasks = [Task(consolidate_merge, similarity_threshold=similarity_threshold)]

        result = await cognee.memify(
            extraction_tasks=extraction_tasks,
            enrichment_tasks=enrichment_tasks,
            dataset=dataset_to_write[0].id,
            data=[{}],
            user=user,
            run_in_background=run_in_background,
        )

    logger.info(
        "Consolidate-Merge memify pipeline completed",
        extra={
            "dataset_id": str(dataset_to_write[0].id),
            "similarity_threshold": similarity_threshold,
        },
    )

    return result
