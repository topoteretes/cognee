from typing import Union
from uuid import UUID

from cognee import memify
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.tasks.memify.global_context_index import (
    extract_global_context_index_input,
    update_global_context_index,
)


def get_global_context_index_memify_tasks(
    max_bucket_size: int = 20,
    placement_distance_threshold: float = 0.5,
    rebuild: bool = False,
):
    return (
        [Task(extract_global_context_index_input)],
        [
            Task(
                update_global_context_index,
                rebuild=rebuild,
                max_bucket_size=max_bucket_size,
                placement_distance_threshold=placement_distance_threshold,
            )
        ],
    )


async def global_context_index_pipeline(
    user: User,
    dataset: Union[str, UUID] = "main_dataset",
    run_in_background: bool = False,
    max_bucket_size: int = 20,
    placement_distance_threshold: float = 0.5,
    rebuild: bool = False,
):
    extraction_tasks, enrichment_tasks = get_global_context_index_memify_tasks(
        max_bucket_size=max_bucket_size,
        placement_distance_threshold=placement_distance_threshold,
        rebuild=rebuild,
    )

    return await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset,
        data=[{}],
        user=user,
        run_in_background=run_in_background,
    )
