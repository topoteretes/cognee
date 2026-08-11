from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def compute_idf_from_counts(
    chunk_count: int,
    entity_chunk_counts: Mapping[str, int],
) -> dict[str, float]:
    """
    Compute entity IDF weights over the summarized chunk population.

    Entities present in every summarized chunk intentionally get weight 0.0 in
    the first graph-bucketing implementation, so ubiquitous entities do not
    drive grouping by themselves.
    """
    if chunk_count <= 0:
        return {}

    idf_weights: dict[str, float] = {}
    for entity_id, entity_chunk_count in entity_chunk_counts.items():
        if entity_chunk_count <= 0:
            continue
        if entity_chunk_count > chunk_count:
            raise ValueError(
                f"entity_chunk_count cannot be greater than chunk_count for entity {entity_id!r}."
            )
        idf_weights[entity_id] = math.log(chunk_count / entity_chunk_count)

    return idf_weights


def entity_weight(entity_id: str, idf_weights: Mapping[str, float]) -> float:
    return idf_weights.get(entity_id, 0.0)


def entities_weight(entity_ids: Iterable[str], idf_weights: Mapping[str, float]) -> float:
    return sum(entity_weight(entity_id, idf_weights) for entity_id in set(entity_ids))


def weighted_jaccard(
    left_entity_ids: Iterable[str],
    right_entity_ids: Iterable[str],
    idf_weights: Mapping[str, float],
) -> float:
    left_entities = set(left_entity_ids)
    right_entities = set(right_entity_ids)

    union_weight = entities_weight(left_entities | right_entities, idf_weights)
    if union_weight == 0:
        return 0.0

    intersection_weight = entities_weight(left_entities & right_entities, idf_weights)
    return intersection_weight / union_weight


def type_similarity(
    left_type_ids: Iterable[str],
    right_type_ids: Iterable[str],
    type_idf_weights: Mapping[str, float],
) -> float:
    """
    Same weighted-Jaccard formula as ``weighted_jaccard``, applied to
    ``EntityType`` ids instead of ``Entity`` ids, over their own IDF weights.
    """
    return weighted_jaccard(left_type_ids, right_type_ids, type_idf_weights)


def combined_similarity(
    entity_score: float,
    type_score: float,
    pattern_score: float,
    entity_weight: float = 1.0,
    type_weight: float = 0.0,
    pattern_weight: float = 0.0,
) -> float:
    """
    Weighted sum of three independent similarity signals (entity, entity-type,
    relationship-pattern). Default weights make this identical to
    ``entity_score`` alone (today's behavior).
    """
    total_weight = entity_weight + type_weight + pattern_weight
    if total_weight == 0:
        return 0.0

    return (
        entity_weight * entity_score + type_weight * type_score + pattern_weight * pattern_score
    ) / total_weight
