from __future__ import annotations

from collections.abc import Mapping

from .bucket_assignment import create_bucket_id
from .idf import entities_weight, weighted_jaccard
from .models import BucketAssignment, SummaryNode


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
        _add_bucket(
            child_ids,
            entities_by_summary_id,
            dataset_id,
            level,
            buckets_to_persist,
            assignments,
        )

    for child_ids in _chunk_ids(sorted(summary.id for summary in misc_summaries), max_bucket_size):
        _add_bucket(
            child_ids,
            entities_by_summary_id,
            dataset_id,
            level,
            buckets_to_persist,
            assignments,
            force_misc=True,
        )

    return buckets_to_persist, assignments


def validate_graph_buckets_can_be_extended(existing_buckets: list[SummaryNode]) -> None:
    for bucket in existing_buckets:
        if bucket.level == 0 and bucket.graph_bucket_entity_ids is None:
            raise ValueError(
                "Graph incremental placement requires existing level-0 buckets to have "
                "graph_bucket_entity_ids. Use rebuild=True to rebuild the index with "
                "graph bucketing."
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


def _build_entity_to_summary_ids(
    summaries: list[SummaryNode],
    entities_by_summary_id: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    entity_to_summary_ids: dict[str, set[str]] = {}
    for summary in summaries:
        for entity_id in entities_by_summary_id.get(summary.id, set()):
            entity_to_summary_ids.setdefault(entity_id, set()).add(summary.id)
    return entity_to_summary_ids


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


def _add_bucket(
    child_ids: list[str],
    entities_by_summary_id: Mapping[str, set[str]],
    dataset_id: str,
    level: int,
    buckets_to_persist: dict[str, SummaryNode],
    assignments: list[BucketAssignment],
    *,
    force_misc: bool = False,
) -> None:
    bucket_id = str(create_bucket_id(dataset_id, level, child_ids))
    bucket = SummaryNode(
        id=bucket_id,
        text="",
        type="GlobalContextSummary",
        level=level,
        is_root=False,
        dataset_id=dataset_id,
        child_ids=set(child_ids),
        graph_bucket_entity_ids=(
            set() if force_misc else _union_entities(child_ids, entities_by_summary_id)
        ),
    )

    buckets_to_persist[bucket.id] = bucket
    assignments.extend(
        BucketAssignment(summary_id=child_id, bucket_id=bucket.id) for child_id in child_ids
    )


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
        summary_ids[index : index + chunk_size]
        for index in range(0, len(summary_ids), chunk_size)
    ]
