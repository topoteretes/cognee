from __future__ import annotations

from collections.abc import Mapping

from ..common import (
    create_bucket_node,
    mark_bucket_for_persistence,
    record_bucket_assignment,
)
from .scoring import entities_weight, weighted_jaccard
from ...ids import create_bucket_id
from ...models import BucketAssignment, SummaryNode


def rebuild_graph_buckets_for_level(
    summaries: list[SummaryNode],
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
    dataset_id: str,
    level: int,
    max_bucket_size: int,
    min_overlap: float,
) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]:
    """
    Build graph-based buckets from in-memory summary/entity inputs.

    This helper is intentionally pure: it does not mutate input summaries, query
    storage, call vector search, or persist bucket state.
    """
    if max_bucket_size < 1:
        raise ValueError("max_bucket_size must be at least 1.")

    entity_summaries, misc_summaries = _partition_summaries(
        summaries, entities_by_summary_id, idf_weights
    )
    entity_to_summary_ids = _build_entity_to_summary_ids(entity_summaries, entities_by_summary_id)

    buckets_to_persist: dict[str, SummaryNode] = {}
    assignments: list[BucketAssignment] = []

    unassigned_ids = {summary.id for summary in entity_summaries}
    for seed in _sort_entity_seeds(entity_summaries, entities_by_summary_id, idf_weights):
        if seed.id not in unassigned_ids:
            continue

        child_ids = _build_entity_bucket_child_ids(
            seed.id,
            unassigned_ids,
            entity_to_summary_ids,
            entities_by_summary_id,
            idf_weights,
            max_bucket_size,
            min_overlap,
        )
        _add_entity_bucket(
            child_ids,
            entities_by_summary_id,
            dataset_id,
            level,
            buckets_to_persist,
            assignments,
        )

    for child_ids in _chunk_ids(sorted(summary.id for summary in misc_summaries), max_bucket_size):
        _add_misc_bucket(
            child_ids,
            dataset_id,
            level,
            buckets_to_persist,
            assignments,
        )

    return buckets_to_persist, assignments


def place_graph_summaries_incrementally(
    new_summaries: list[SummaryNode],
    existing_buckets: list[SummaryNode],
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
    dataset_id: str,
    level: int,
    max_bucket_size: int,
    min_overlap: float,
) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]:
    if level != 0:
        raise ValueError("Graph incremental placement is only supported for level 0.")
    if max_bucket_size < 1:
        raise ValueError("max_bucket_size must be at least 1.")

    validate_graph_buckets_can_be_extended(existing_buckets)

    buckets_by_id = {bucket.id: bucket for bucket in existing_buckets}
    buckets_to_persist = _normalize_existing_graph_buckets(
        existing_buckets,
        entities_by_summary_id,
        idf_weights,
    )
    entity_to_bucket_ids = _build_entity_to_bucket_ids(existing_buckets)
    misc_bucket_ids = _build_misc_bucket_ids(existing_buckets)
    assignments: list[BucketAssignment] = []

    for summary in sorted(new_summaries, key=lambda item: item.id):
        summary_entity_ids = entities_by_summary_id.get(summary.id, set())
        if not _has_positive_entity_weight(summary_entity_ids, idf_weights):
            _place_misc_summary(
                summary,
                buckets_by_id,
                misc_bucket_ids,
                entities_by_summary_id,
                dataset_id,
                level,
                max_bucket_size,
                buckets_to_persist,
                assignments,
            )
            continue

        bucket = _choose_existing_graph_bucket(
            summary_entity_ids,
            entity_to_bucket_ids,
            buckets_by_id,
            idf_weights,
            max_bucket_size,
            min_overlap,
        )
        if bucket is None:
            bucket = _create_new_graph_bucket(
                summary,
                entities_by_summary_id,
                dataset_id,
                level,
                buckets_by_id,
                buckets_to_persist,
                assignments,
            )
        else:
            _assign_summary_to_graph_bucket(
                summary,
                bucket,
                summary_entity_ids,
                buckets_to_persist,
                assignments,
            )

        _index_bucket_entities(bucket, entity_to_bucket_ids)

    return buckets_to_persist, assignments


def validate_graph_buckets_can_be_extended(existing_buckets: list[SummaryNode]) -> None:
    for bucket in existing_buckets:
        if bucket.level == 0 and bucket.graph_bucket_entity_ids is None:
            raise ValueError(
                "Graph incremental placement requires existing level-0 buckets to have "
                "graph_bucket_entity_ids. Use rebuild=True to rebuild the index with "
                "graph bucketing."
            )


def validate_vector_buckets_can_be_extended(existing_buckets: list[SummaryNode]) -> None:
    for bucket in existing_buckets:
        if bucket.level == 0 and bucket.graph_bucket_entity_ids is not None:
            raise ValueError(
                "Vector incremental placement cannot extend graph-built level-0 buckets. "
                "Use rebuild=True to rebuild the index with vector bucketing."
            )


def _partition_summaries(
    summaries: list[SummaryNode],
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
) -> tuple[list[SummaryNode], list[SummaryNode]]:
    entity_summaries: list[SummaryNode] = []
    misc_summaries: list[SummaryNode] = []

    for summary in summaries:
        entity_ids = entities_by_summary_id.get(summary.id, set())
        if entity_ids and entities_weight(entity_ids, idf_weights) > 0:
            entity_summaries.append(summary)
        else:
            misc_summaries.append(summary)

    return entity_summaries, misc_summaries


def _normalize_existing_graph_buckets(
    existing_buckets: list[SummaryNode],
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
) -> dict[str, SummaryNode]:
    buckets_to_persist: dict[str, SummaryNode] = {}

    for bucket in existing_buckets:
        child_entity_ids = _union_entities(sorted(bucket.child_ids), entities_by_summary_id)
        normalized_entity_ids = (
            child_entity_ids
            if _has_positive_entity_weight(child_entity_ids, idf_weights)
            else set()
        )
        if bucket.graph_bucket_entity_ids != normalized_entity_ids:
            bucket.graph_bucket_entity_ids = normalized_entity_ids
            buckets_to_persist[bucket.id] = bucket

    return buckets_to_persist


def _has_positive_entity_weight(
    entity_ids: set[str],
    idf_weights: Mapping[str, float],
) -> bool:
    return bool(entity_ids) and entities_weight(entity_ids, idf_weights) > 0


def _build_entity_to_bucket_ids(
    buckets: list[SummaryNode],
) -> dict[str, set[str]]:
    entity_to_bucket_ids: dict[str, set[str]] = {}
    for bucket in buckets:
        _index_bucket_entities(bucket, entity_to_bucket_ids)
    return entity_to_bucket_ids


def _index_bucket_entities(
    bucket: SummaryNode,
    entity_to_bucket_ids: dict[str, set[str]],
) -> None:
    if bucket.graph_bucket_entity_ids is None:
        return

    for entity_id in bucket.graph_bucket_entity_ids:
        entity_to_bucket_ids.setdefault(entity_id, set()).add(bucket.id)


def _build_misc_bucket_ids(
    buckets: list[SummaryNode],
) -> list[str]:
    return sorted(
        bucket.id
        for bucket in buckets
        if bucket.graph_bucket_entity_ids is not None and not bucket.graph_bucket_entity_ids
    )


def _build_entity_to_summary_ids(
    summaries: list[SummaryNode],
    entities_by_summary_id: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    entity_to_summary_ids: dict[str, set[str]] = {}
    for summary in summaries:
        for entity_id in entities_by_summary_id.get(summary.id, set()):
            entity_to_summary_ids.setdefault(entity_id, set()).add(summary.id)
    return entity_to_summary_ids


def _choose_existing_graph_bucket(
    summary_entity_ids: set[str],
    entity_to_bucket_ids: Mapping[str, set[str]],
    buckets_by_id: Mapping[str, SummaryNode],
    idf_weights: Mapping[str, float],
    max_bucket_size: int,
    min_overlap: float,
) -> SummaryNode | None:
    candidate_bucket_ids: set[str] = set()
    for entity_id in summary_entity_ids:
        candidate_bucket_ids.update(entity_to_bucket_ids.get(entity_id, set()))

    scored_candidates: list[tuple[float, int, str, SummaryNode]] = []
    for bucket_id in sorted(candidate_bucket_ids):
        bucket = buckets_by_id[bucket_id]
        if len(bucket.child_ids) >= max_bucket_size:
            continue

        score = weighted_jaccard(
            summary_entity_ids,
            bucket.graph_bucket_entity_ids or set(),
            idf_weights,
        )
        if score < min_overlap:
            continue
        scored_candidates.append((-score, len(bucket.child_ids), bucket.id, bucket))

    if not scored_candidates:
        return None

    return sorted(scored_candidates, key=lambda item: (item[0], item[1], item[2]))[0][3]


def _place_misc_summary(
    summary: SummaryNode,
    buckets_by_id: dict[str, SummaryNode],
    misc_bucket_ids: list[str],
    entities_by_summary_id: Mapping[str, set[str]],
    dataset_id: str,
    level: int,
    max_bucket_size: int,
    buckets_to_persist: dict[str, SummaryNode],
    assignments: list[BucketAssignment],
) -> None:
    bucket = _first_available_misc_bucket(misc_bucket_ids, buckets_by_id, max_bucket_size)
    if bucket is None:
        bucket = _create_new_misc_bucket(
            summary,
            dataset_id,
            level,
            buckets_by_id,
            buckets_to_persist,
            assignments,
        )
        misc_bucket_ids.append(bucket.id)
        misc_bucket_ids.sort()
        return

    _assign_summary_to_misc_bucket(
        summary,
        bucket,
        buckets_to_persist,
        assignments,
    )


def _first_available_misc_bucket(
    misc_bucket_ids: list[str],
    buckets_by_id: Mapping[str, SummaryNode],
    max_bucket_size: int,
) -> SummaryNode | None:
    for bucket_id in misc_bucket_ids:
        bucket = buckets_by_id[bucket_id]
        if len(bucket.child_ids) < max_bucket_size:
            return bucket
    return None


def _create_new_graph_bucket(
    summary: SummaryNode,
    entities_by_summary_id: Mapping[str, set[str]],
    dataset_id: str,
    level: int,
    buckets_by_id: dict[str, SummaryNode],
    buckets_to_persist: dict[str, SummaryNode],
    assignments: list[BucketAssignment],
) -> SummaryNode:
    _add_entity_bucket(
        [summary.id],
        entities_by_summary_id,
        dataset_id,
        level,
        buckets_to_persist,
        assignments,
    )
    bucket_id = str(create_bucket_id(dataset_id, level, [summary.id]))
    bucket = buckets_to_persist[bucket_id]
    buckets_by_id[bucket.id] = bucket
    return bucket


def _create_new_misc_bucket(
    summary: SummaryNode,
    dataset_id: str,
    level: int,
    buckets_by_id: dict[str, SummaryNode],
    buckets_to_persist: dict[str, SummaryNode],
    assignments: list[BucketAssignment],
) -> SummaryNode:
    _add_misc_bucket(
        [summary.id],
        dataset_id,
        level,
        buckets_to_persist,
        assignments,
    )
    bucket_id = str(create_bucket_id(dataset_id, level, [summary.id]))
    bucket = buckets_to_persist[bucket_id]
    buckets_by_id[bucket.id] = bucket
    return bucket


def _assign_summary_to_graph_bucket(
    summary: SummaryNode,
    bucket: SummaryNode,
    summary_entity_ids: set[str],
    buckets_to_persist: dict[str, SummaryNode],
    assignments: list[BucketAssignment],
) -> None:
    bucket.child_ids.add(summary.id)
    if bucket.graph_bucket_entity_ids is not None:
        bucket.graph_bucket_entity_ids.update(summary_entity_ids)

    mark_bucket_for_persistence(buckets_to_persist, bucket)
    record_bucket_assignment(assignments, summary.id, bucket.id)


def _assign_summary_to_misc_bucket(
    summary: SummaryNode,
    bucket: SummaryNode,
    buckets_to_persist: dict[str, SummaryNode],
    assignments: list[BucketAssignment],
) -> None:
    bucket.child_ids.add(summary.id)
    bucket.graph_bucket_entity_ids = set()
    mark_bucket_for_persistence(buckets_to_persist, bucket)
    record_bucket_assignment(assignments, summary.id, bucket.id)


def _sort_entity_seeds(
    summaries: list[SummaryNode],
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
) -> list[SummaryNode]:
    return sorted(
        summaries,
        key=lambda summary: (
            -entities_weight(entities_by_summary_id.get(summary.id, set()), idf_weights),
            summary.id,
        ),
    )


def _build_entity_bucket_child_ids(
    seed_id: str,
    unassigned_ids: set[str],
    entity_to_summary_ids: Mapping[str, set[str]],
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
    max_bucket_size: int,
    min_overlap: float,
) -> list[str]:
    child_ids = [seed_id]
    unassigned_ids.remove(seed_id)
    bucket_entity_ids = set(entities_by_summary_id.get(seed_id, set()))

    while len(child_ids) < max_bucket_size:
        candidate_id = _choose_best_candidate(
            bucket_entity_ids,
            unassigned_ids,
            entity_to_summary_ids,
            entities_by_summary_id,
            idf_weights,
            min_overlap,
        )
        if candidate_id is None:
            break

        child_ids.append(candidate_id)
        unassigned_ids.remove(candidate_id)
        bucket_entity_ids.update(entities_by_summary_id.get(candidate_id, set()))

    return child_ids


def _choose_best_candidate(
    bucket_entity_ids: set[str],
    unassigned_ids: set[str],
    entity_to_summary_ids: Mapping[str, set[str]],
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
    min_overlap: float,
) -> str | None:
    candidate_ids = set()
    for entity_id in bucket_entity_ids:
        candidate_ids.update(entity_to_summary_ids.get(entity_id, set()))
    candidate_ids &= unassigned_ids

    best_id: str | None = None
    best_score = float("-inf")
    for candidate_id in sorted(candidate_ids):
        score = weighted_jaccard(
            entities_by_summary_id.get(candidate_id, set()),
            bucket_entity_ids,
            idf_weights,
        )
        if score < min_overlap:
            continue
        if score > best_score:
            best_id = candidate_id
            best_score = score

    return best_id


def _add_entity_bucket(
    child_ids: list[str],
    entities_by_summary_id: Mapping[str, set[str]],
    dataset_id: str,
    level: int,
    buckets_to_persist: dict[str, SummaryNode],
    assignments: list[BucketAssignment],
) -> None:
    _add_bucket_with_entities(
        child_ids,
        _union_entities(child_ids, entities_by_summary_id),
        dataset_id,
        level,
        buckets_to_persist,
        assignments,
    )


def _add_misc_bucket(
    child_ids: list[str],
    dataset_id: str,
    level: int,
    buckets_to_persist: dict[str, SummaryNode],
    assignments: list[BucketAssignment],
) -> None:
    _add_bucket_with_entities(
        child_ids,
        set(),
        dataset_id,
        level,
        buckets_to_persist,
        assignments,
    )


def _add_bucket_with_entities(
    child_ids: list[str],
    graph_bucket_entity_ids: set[str],
    dataset_id: str,
    level: int,
    buckets_to_persist: dict[str, SummaryNode],
    assignments: list[BucketAssignment],
) -> None:
    bucket = create_bucket_node(
        child_ids,
        dataset_id,
        level,
        graph_bucket_entity_ids=graph_bucket_entity_ids,
    )
    mark_bucket_for_persistence(buckets_to_persist, bucket)
    for child_id in child_ids:
        record_bucket_assignment(assignments, child_id, bucket.id)


def _union_entities(
    child_ids: list[str],
    entities_by_summary_id: Mapping[str, set[str]],
) -> set[str]:
    entity_ids: set[str] = set()
    for child_id in child_ids:
        entity_ids.update(entities_by_summary_id.get(child_id, set()))
    return entity_ids


def _chunk_ids(summary_ids: list[str], chunk_size: int) -> list[list[str]]:
    return [
        summary_ids[index : index + chunk_size] for index in range(0, len(summary_ids), chunk_size)
    ]
