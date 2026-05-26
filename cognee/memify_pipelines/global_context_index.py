from typing import Union
from uuid import UUID

from cognee import memify
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.tasks.memify.global_context_index import (
    extract_global_context_index_input,
    update_global_context_index,
)
from cognee.tasks.memify.global_context_index.bucketing_strategy import BucketingStrategyName


def get_global_context_index_memify_tasks(
    max_bucket_size: int = 20,
    placement_distance_threshold: float = 0.5,
    rebuild: bool = False,
    bucketing_strategy: BucketingStrategyName = "vector",
    min_overlap: float = 0.05,
):
    """
    Build the task pair for the explicit global context index pipeline.

    ``bucketing_strategy="vector"`` is the default and uses
    ``placement_distance_threshold`` for level-0 and upper-level placement.
    ``bucketing_strategy="graph"`` is experimental, uses ``min_overlap`` for
    level-0 graph placement, and falls back to vector bucketing for levels
    above 0. Switching between vector-built and graph-built indexes requires
    ``rebuild=True``.
    """
    return (
        [Task(extract_global_context_index_input)],
        [
            Task(
                update_global_context_index,
                rebuild=rebuild,
                max_bucket_size=max_bucket_size,
                placement_distance_threshold=placement_distance_threshold,
                bucketing_strategy=bucketing_strategy,
                min_overlap=min_overlap,
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
    bucketing_strategy: BucketingStrategyName = "vector",
    min_overlap: float = 0.05,
):
    """
    Build or update the global context index for a dataset.

    Vector bucketing is the default. Graph bucketing is experimental and only
    applies to level-0 buckets; upper levels remain vector-based. Use
    ``rebuild=True`` when switching between vector and graph strategies.
    """
    extraction_tasks, enrichment_tasks = get_global_context_index_memify_tasks(
        max_bucket_size=max_bucket_size,
        placement_distance_threshold=placement_distance_threshold,
        rebuild=rebuild,
        bucketing_strategy=bucketing_strategy,
        min_overlap=min_overlap,
    )

    return await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset,
        data=[{}],
        user=user,
        run_in_background=run_in_background,
    )
