from __future__ import annotations

from typing import Any

from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.summarization.models import GlobalContextSummary

from .bucket_assignment import assign_items_to_buckets, create_buckets_for_level
from .constants import GLOBAL_CONTEXT_SUMMARY_COLLECTION, TEXT_SUMMARY_COLLECTION
from .models import BucketAssignment, SummaryNode
from .persistence import persist_context_summaries
from .summary_generation import (
    build_global_context_summary_datapoint,
    generate_bucket_summary_datapoints,
)


def group_buckets_by_level(buckets: list[SummaryNode]) -> dict[int, list[SummaryNode]]:
    by_level: dict[int, list[SummaryNode]] = {}
    for bucket in buckets:
        if bucket.level is None:
            continue
        by_level.setdefault(bucket.level, []).append(bucket)
    return by_level


async def build_context_index_level(
    items: list[SummaryNode],
    items_for_cluster: list[SummaryNode],
    existing: list[SummaryNode],
    children_by_id: dict[str, SummaryNode],
    level: int,
    dataset_id: str,
    vector_engine: Any,
    max_bucket_size: int,
    placement_distance_threshold: float,
    ctx: PipelineContext | None,
) -> tuple[
    dict[str, SummaryNode],
    list[BucketAssignment],
    list[GlobalContextSummary],
]:
    source_collection = TEXT_SUMMARY_COLLECTION if level == 0 else GLOBAL_CONTEXT_SUMMARY_COLLECTION

    if existing or level == 0:
        buckets_to_persist, assignments = await assign_items_to_buckets(
            items,
            existing,
            level,
            dataset_id,
            vector_engine,
            source_collection,
            max_bucket_size,
            placement_distance_threshold,
        )
    else:
        buckets_to_persist, assignments = await create_buckets_for_level(
            items_for_cluster,
            level,
            dataset_id,
            vector_engine,
            source_collection,
            max_bucket_size,
        )

    bucket_datapoints = await generate_bucket_summary_datapoints(
        list(buckets_to_persist.values()), children_by_id, dataset_id
    )
    await persist_context_summaries(bucket_datapoints, ctx)
    return buckets_to_persist, assignments, bucket_datapoints


async def build_context_index(
    new_text_summaries: list[SummaryNode],
    text_summaries_all: list[SummaryNode],
    buckets_by_level: dict[int, list[SummaryNode]],
    existing_root: SummaryNode | None,
    dataset_id: str,
    vector_engine: Any,
    max_bucket_size: int,
    placement_distance_threshold: float,
    ctx: PipelineContext | None,
) -> tuple[list[GlobalContextSummary], list[BucketAssignment]]:
    """
    Bottom-up global context index build starting from new TextSummaries.

    Each iteration runs the per-level pattern: place changed items into the
    existing level, or cluster all items at the previous level when the target
    level does not exist yet. The loop stops once the topmost non-root level
    fits in the root's capacity. Root is regenerated when anything below
    changed or when no root exists yet.
    """
    all_assignments: list[BucketAssignment] = []
    all_datapoints: list[GlobalContextSummary] = []

    items_changed: list[SummaryNode] = list(new_text_summaries)
    items_all: list[SummaryNode] = list(text_summaries_all)
    children_by_id: dict[str, SummaryNode] = {s.id: s for s in text_summaries_all}

    level = 0
    top_level_buckets: list[SummaryNode] = []

    while True:
        existing = buckets_by_level.get(level, [])
        buckets_to_persist, assignments, datapoints = await build_context_index_level(
            items_changed,
            items_all,
            existing,
            children_by_id,
            level,
            dataset_id,
            vector_engine,
            max_bucket_size,
            placement_distance_threshold,
            ctx,
        )
        all_assignments.extend(assignments)
        all_datapoints.extend(datapoints)

        existing_ids = {bucket.id for bucket in existing}
        new_at_level = [
            bucket for bucket in buckets_to_persist.values() if bucket.id not in existing_ids
        ]
        buckets_at_level = list(existing) + new_at_level

        if len(buckets_at_level) <= max_bucket_size:
            top_level_buckets = buckets_at_level
            break

        items_changed = list(buckets_to_persist.values())
        items_all = buckets_at_level
        children_by_id = {bucket.id: bucket for bucket in buckets_at_level}
        level += 1

    if top_level_buckets and (all_assignments or existing_root is None):
        root_datapoint = await build_global_context_summary_datapoint(
            top_level_buckets, dataset_id, level + 1
        )
        await persist_context_summaries([root_datapoint], ctx)
        all_datapoints.append(root_datapoint)

        root_id_str = str(root_datapoint.id)
        for bucket in top_level_buckets:
            if bucket.global_context_bucket_id == root_id_str:
                continue
            all_assignments.append(BucketAssignment(summary_id=bucket.id, bucket_id=root_id_str))
            bucket.global_context_bucket_id = root_id_str

    return all_datapoints, all_assignments
