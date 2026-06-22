from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.summarization.models import GlobalContextSummary

from .bucketing.graph.placement import (
    place_graph_summaries_incrementally,
    rebuild_graph_buckets_for_level,
)
from .bucketing.vector.placement import assign_items_to_buckets, create_buckets_for_level
from .bucketing_strategy import BucketingStrategyName
from .constants import GLOBAL_CONTEXT_SUMMARY_COLLECTION, TEXT_SUMMARY_COLLECTION
from .models import BucketAssignment, SummaryNode
from .persist import persist_context_summaries
from .summarize import (
    build_global_context_summary_datapoint,
    generate_bucket_summary_datapoints,
)


@dataclass(frozen=True)
class BuildOptions:
    dataset_id: str
    vector_engine: Any
    max_bucket_size: int
    placement_distance_threshold: float
    bucketing_strategy: BucketingStrategyName
    min_overlap: float
    entities_by_summary_id: Mapping[str, set[str]]
    idf_weights: Mapping[str, float]
    ctx: PipelineContext | None


def group_buckets_by_level(buckets: list[SummaryNode]) -> dict[int, list[SummaryNode]]:
    by_level: dict[int, list[SummaryNode]] = {}
    for bucket in buckets:
        if bucket.level is None:
            continue
        by_level.setdefault(bucket.level, []).append(bucket)
    return by_level


def apply_bucket_assignments(
    assignments: list[BucketAssignment],
    children_by_id: dict[str, SummaryNode],
) -> None:
    for assignment in assignments:
        child = children_by_id.get(assignment.child_id)
        if child is not None:
            child.global_context_bucket_id = assignment.parent_id


def source_collection_for_level(level: int) -> str:
    if level == 0:
        return TEXT_SUMMARY_COLLECTION
    return GLOBAL_CONTEXT_SUMMARY_COLLECTION


def place_graph_items(
    changed_items: list[SummaryNode],
    all_items: list[SummaryNode],
    existing_buckets: list[SummaryNode],
    level: int,
    options: BuildOptions,
) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]:
    if existing_buckets:
        return place_graph_summaries_incrementally(
            changed_items,
            existing_buckets,
            options.entities_by_summary_id,
            options.idf_weights,
            dataset_id=options.dataset_id,
            level=level,
            max_bucket_size=options.max_bucket_size,
            min_overlap=options.min_overlap,
        )

    return rebuild_graph_buckets_for_level(
        all_items,
        options.entities_by_summary_id,
        options.idf_weights,
        dataset_id=options.dataset_id,
        level=level,
        max_bucket_size=options.max_bucket_size,
        min_overlap=options.min_overlap,
    )


async def place_vector_items(
    changed_items: list[SummaryNode],
    all_items: list[SummaryNode],
    existing_buckets: list[SummaryNode],
    level: int,
    options: BuildOptions,
) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]:
    source_collection = source_collection_for_level(level)
    if existing_buckets or level == 0:
        return await assign_items_to_buckets(
            changed_items,
            existing_buckets,
            level,
            options.dataset_id,
            options.vector_engine,
            source_collection,
            options.max_bucket_size,
            options.placement_distance_threshold,
        )

    return await create_buckets_for_level(
        all_items,
        level,
        options.dataset_id,
        options.vector_engine,
        source_collection,
        options.max_bucket_size,
    )


async def place_items_for_level(
    changed_items: list[SummaryNode],
    all_items: list[SummaryNode],
    existing_buckets: list[SummaryNode],
    level: int,
    options: BuildOptions,
) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]:
    if options.bucketing_strategy == "graph" and level == 0:
        return place_graph_items(changed_items, all_items, existing_buckets, level, options)

    return await place_vector_items(changed_items, all_items, existing_buckets, level, options)


async def build_and_persist_level(
    changed_items: list[SummaryNode],
    all_items: list[SummaryNode],
    existing_buckets: list[SummaryNode],
    children_by_id: dict[str, SummaryNode],
    level: int,
    options: BuildOptions,
) -> tuple[dict[str, SummaryNode], list[BucketAssignment], list[GlobalContextSummary]]:
    buckets_to_persist, assignments = await place_items_for_level(
        changed_items,
        all_items,
        existing_buckets,
        level,
        options,
    )
    apply_bucket_assignments(assignments, children_by_id)

    bucket_datapoints = await generate_bucket_summary_datapoints(
        list(buckets_to_persist.values()), children_by_id, options.dataset_id
    )
    await persist_context_summaries(bucket_datapoints, options.ctx)
    return buckets_to_persist, assignments, bucket_datapoints


def merge_existing_and_new_buckets(
    existing_buckets: list[SummaryNode],
    buckets_to_persist: dict[str, SummaryNode],
) -> list[SummaryNode]:
    existing_ids = {bucket.id for bucket in existing_buckets}
    new_buckets = [
        bucket for bucket in buckets_to_persist.values() if bucket.id not in existing_ids
    ]
    return list(existing_buckets) + new_buckets


def should_update_root(
    top_level_buckets: list[SummaryNode],
    assignments: list[BucketAssignment],
    existing_root: SummaryNode | None,
) -> bool:
    return bool(top_level_buckets) and (bool(assignments) or existing_root is None)


async def build_and_persist_root(
    top_level_buckets: list[SummaryNode],
    root_level: int,
    options: BuildOptions,
) -> tuple[GlobalContextSummary, list[BucketAssignment]]:
    root_datapoint = await build_global_context_summary_datapoint(
        top_level_buckets, options.dataset_id, root_level
    )
    await persist_context_summaries([root_datapoint], options.ctx)

    root_id = str(root_datapoint.id)
    assignments: list[BucketAssignment] = []
    for bucket in top_level_buckets:
        if bucket.global_context_bucket_id == root_id:
            continue
        assignments.append(BucketAssignment(child_id=bucket.id, parent_id=root_id))
        bucket.global_context_bucket_id = root_id

    return root_datapoint, assignments


async def build_context_index(
    new_text_summaries: list[SummaryNode],
    text_summaries_all: list[SummaryNode],
    buckets_by_level: dict[int, list[SummaryNode]],
    existing_root: SummaryNode | None,
    dataset_id: str,
    vector_engine: Any,
    max_bucket_size: int,
    placement_distance_threshold: float,
    bucketing_strategy: BucketingStrategyName = "vector",
    min_overlap: float = 0.05,
    entities_by_summary_id: dict[str, set[str]] | None = None,
    idf_weights: dict[str, float] | None = None,
    ctx: PipelineContext | None = None,
) -> tuple[list[GlobalContextSummary], list[BucketAssignment]]:
    """
    Bottom-up global context index build starting from new TextSummaries.

    Each iteration runs the per-level pattern: place changed items into the
    existing level, or cluster all items at the previous level when the target
    level does not exist yet. The loop stops once the topmost non-root level
    fits in the root's capacity. Root is regenerated when anything below
    changed or when no root exists yet.
    """
    options = BuildOptions(
        dataset_id=dataset_id,
        vector_engine=vector_engine,
        max_bucket_size=max_bucket_size,
        placement_distance_threshold=placement_distance_threshold,
        bucketing_strategy=bucketing_strategy,
        min_overlap=min_overlap,
        entities_by_summary_id=entities_by_summary_id or {},
        idf_weights=idf_weights or {},
        ctx=ctx,
    )

    items_changed: list[SummaryNode] = list(new_text_summaries)
    items_all: list[SummaryNode] = list(text_summaries_all)
    children_by_id: dict[str, SummaryNode] = {s.id: s for s in text_summaries_all}
    level = 0
    top_level_buckets: list[SummaryNode] = []
    all_assignments: list[BucketAssignment] = []
    all_datapoints: list[GlobalContextSummary] = []

    while True:
        existing = buckets_by_level.get(level, [])
        buckets_to_persist, assignments, datapoints = await build_and_persist_level(
            items_changed,
            items_all,
            existing,
            children_by_id,
            level,
            options,
        )
        all_assignments.extend(assignments)
        all_datapoints.extend(datapoints)

        buckets_at_level = merge_existing_and_new_buckets(existing, buckets_to_persist)
        if len(buckets_at_level) <= max_bucket_size:
            top_level_buckets = buckets_at_level
            break

        items_changed = list(buckets_to_persist.values())
        items_all = buckets_at_level
        children_by_id = {bucket.id: bucket for bucket in buckets_at_level}
        level += 1

    if should_update_root(top_level_buckets, all_assignments, existing_root):
        root_datapoint, root_assignments = await build_and_persist_root(
            top_level_buckets,
            root_level=level + 1,
            options=options,
        )
        all_datapoints.append(root_datapoint)
        all_assignments.extend(root_assignments)

    return all_datapoints, all_assignments
