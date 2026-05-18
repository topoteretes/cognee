from __future__ import annotations

import asyncio
from typing import Any
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

from .models import BucketAssignment, SummaryNode


def create_bucket_id(dataset_id: str, level: int, child_ids: list[str]) -> UUID:
    child_key = ",".join(sorted(child_ids))
    return uuid5(NAMESPACE_OID, f"GlobalContextSummary:{dataset_id}:{level}:{child_key}")


def create_root_summary_id(dataset_id: str) -> UUID:
    return uuid5(NAMESPACE_OID, f"GlobalContextSummary:{dataset_id}:root")


def create_bucket_for_item(
    summary: SummaryNode,
    dataset_id: str,
    level: int = 0,
) -> SummaryNode:
    return SummaryNode(
        id=str(create_bucket_id(dataset_id, level, [summary.id])),
        text="",
        type="GlobalContextSummary",
        level=level,
        is_root=False,
        dataset_id=dataset_id,
        child_ids={summary.id},
    )


async def prefetch_nearest_neighbors(
    items: list[SummaryNode],
    placed_child_count: int,
    vector_engine: Any,
    collection_name: str,
) -> dict[str, list]:
    """
    Concurrently fetch each item's nearest neighbors from the source
    collection. Vector store contents are stable during a sweep iteration,
    so the cached results remain valid for the placement loop.
    """
    if not items:
        return {}

    in_scope_count = placed_child_count + len(items)
    limit = max(50, in_scope_count * 2)

    async def _search(item: SummaryNode) -> tuple[str, list]:
        try:
            results = await vector_engine.search(
                collection_name,
                item.text,
                limit=limit,
                include_payload=False,
            )
        except CollectionNotFoundError:
            results = []
        return item.id, results

    pairs = await asyncio.gather(*(_search(item) for item in items))
    return dict(pairs)


def build_child_to_bucket_index(
    buckets_by_id: dict[str, SummaryNode],
) -> dict[str, str]:
    return {
        child_id: bucket.id for bucket in buckets_by_id.values() for child_id in bucket.child_ids
    }


def group_distances_by_bucket(
    nearest_results: list,
    child_id_to_bucket_id: dict[str, str],
) -> dict[str, list[float]]:
    distances_by_bucket: dict[str, list[float]] = {}
    for result in nearest_results:
        bucket_id = child_id_to_bucket_id.get(str(getattr(result, "id", "")))
        distance = getattr(result, "score", None)
        if bucket_id is None or distance is None:
            continue
        distances_by_bucket.setdefault(bucket_id, []).append(distance)
    return distances_by_bucket


def choose_best_bucket(
    distances_by_bucket: dict[str, list[float]],
    buckets_by_id: dict[str, SummaryNode],
    max_bucket_size: int,
    placement_distance_threshold: float,
) -> SummaryNode | None:
    best_bucket: SummaryNode | None = None
    best_mean = float("inf")

    for bucket_id, distances in distances_by_bucket.items():
        bucket = buckets_by_id[bucket_id]
        # Skip buckets whose children weren't all in the prefetched window:
        # a missing child is farther than top-K and cannot be scored fairly.
        if len(distances) != len(bucket.child_ids):
            continue
        if len(bucket.child_ids) >= max_bucket_size:
            continue
        mean = sum(distances) / len(distances)
        if mean > placement_distance_threshold:
            continue
        is_better = mean < best_mean or (
            mean == best_mean
            and best_bucket is not None
            and len(bucket.child_ids) < len(best_bucket.child_ids)
        )
        if is_better:
            best_bucket = bucket
            best_mean = mean

    return best_bucket


def choose_existing_bucket_for_item(
    buckets_by_id: dict[str, SummaryNode],
    nearest_results: list,
    max_bucket_size: int,
    placement_distance_threshold: float,
) -> SummaryNode | None:
    if not buckets_by_id or not nearest_results:
        return None

    child_id_to_bucket_id = build_child_to_bucket_index(buckets_by_id)
    if not child_id_to_bucket_id:
        return None

    distances_by_bucket = group_distances_by_bucket(nearest_results, child_id_to_bucket_id)
    return choose_best_bucket(
        distances_by_bucket,
        buckets_by_id,
        max_bucket_size,
        placement_distance_threshold,
    )


async def assign_items_to_buckets(
    items: list[SummaryNode],
    existing_buckets: list[SummaryNode],
    level: int,
    dataset_id: str,
    vector_engine: Any,
    source_collection: str,
    max_bucket_size: int,
    placement_distance_threshold: float,
) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]:
    """
    Place items into buckets at this level: extend an existing parent or
    create a new bucket. Items already pointing to a parent flag that parent
    for regeneration so its summary refreshes.
    """
    buckets_by_id = {bucket.id: bucket for bucket in existing_buckets}
    placed_child_count = sum(len(bucket.child_ids) for bucket in existing_buckets)

    needs_placement = [item for item in items if not item.global_context_bucket_id]
    items_with_parent = [item for item in items if item.global_context_bucket_id]

    nearest_by_id = await prefetch_nearest_neighbors(
        needs_placement, placed_child_count, vector_engine, source_collection
    )

    buckets_to_persist: dict[str, SummaryNode] = {}
    assignments: list[BucketAssignment] = []

    for item in items_with_parent:
        parent = buckets_by_id.get(item.global_context_bucket_id)
        if parent is not None:
            buckets_to_persist[parent.id] = parent

    for item in needs_placement:
        bucket = choose_existing_bucket_for_item(
            buckets_by_id,
            nearest_by_id.get(item.id, []),
            max_bucket_size,
            placement_distance_threshold,
        )
        if bucket is None:
            bucket = create_bucket_for_item(item, dataset_id, level=level)
            buckets_by_id[bucket.id] = bucket
        else:
            bucket.child_ids.add(item.id)

        item.global_context_bucket_id = bucket.id
        buckets_to_persist[bucket.id] = bucket
        assignments.append(BucketAssignment(summary_id=item.id, bucket_id=bucket.id))

    return buckets_to_persist, assignments


async def create_buckets_for_level(
    items: list[SummaryNode],
    level: int,
    dataset_id: str,
    vector_engine: Any,
    source_collection: str,
    max_bucket_size: int,
) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]:
    """
    Greedy nearest-neighbor clustering. Used when first creating a level above
    level 0: pick a seed from items in input order, attach its nearest
    unclustered neighbors, then top up by input order.
    """
    if not items:
        return {}, []

    nearest_by_id = await prefetch_nearest_neighbors(
        items, len(items), vector_engine, source_collection
    )

    item_by_id = {item.id: item for item in items}
    unclustered = {item.id for item in items}
    buckets_to_persist: dict[str, SummaryNode] = {}
    assignments: list[BucketAssignment] = []

    def attach(bucket: SummaryNode, candidate_id: str) -> None:
        bucket.child_ids.add(candidate_id)
        unclustered.discard(candidate_id)
        child = item_by_id[candidate_id]
        child.global_context_bucket_id = bucket.id
        assignments.append(BucketAssignment(summary_id=candidate_id, bucket_id=bucket.id))

    for seed in items:
        if seed.id not in unclustered:
            continue

        bucket = create_bucket_for_item(seed, dataset_id, level=level)
        unclustered.discard(seed.id)
        seed.global_context_bucket_id = bucket.id
        assignments.append(BucketAssignment(summary_id=seed.id, bucket_id=bucket.id))

        for result in nearest_by_id.get(seed.id, []):
            if len(bucket.child_ids) >= max_bucket_size:
                break
            candidate_id = str(getattr(result, "id", ""))
            if candidate_id == seed.id or candidate_id not in unclustered:
                continue
            attach(bucket, candidate_id)

        if len(bucket.child_ids) < max_bucket_size:
            for fallback in items:
                if len(bucket.child_ids) >= max_bucket_size:
                    break
                if fallback.id not in unclustered:
                    continue
                attach(bucket, fallback.id)

        buckets_to_persist[bucket.id] = bucket

    return buckets_to_persist, assignments
