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


def type_ids_for_entities(
    entity_ids: Iterable[str],
    entity_type_by_entity_id: Mapping[str, str],
) -> set[str]:
    return {
        entity_type_by_entity_id[entity_id]
        for entity_id in entity_ids
        if entity_id in entity_type_by_entity_id
    }


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


def cosine_distance(left_vector: Iterable[float], right_vector: Iterable[float]) -> float:
    left = list(left_vector)
    right = list(right_vector)

    dot_product = sum(
        left_component * right_component for left_component, right_component in zip(left, right)
    )
    left_norm = math.sqrt(sum(component * component for component in left))
    right_norm = math.sqrt(sum(component * component for component in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0

    return 1.0 - dot_product / (left_norm * right_norm)


def relationship_match(
    left_relationship_name: str,
    right_relationship_name: str,
    edge_type_embeddings: Mapping[str, list[float]],
    distance_threshold: float,
) -> bool:
    """
    Two relationship names count as "the same relationship" if they're
    identical, or if their embeddings are close enough (below
    ``distance_threshold``, measured on real synonym/non-synonym pairs, not
    assumed).
    """
    if left_relationship_name == right_relationship_name:
        return True

    left_vector = edge_type_embeddings.get(left_relationship_name)
    right_vector = edge_type_embeddings.get(right_relationship_name)
    if left_vector is None or right_vector is None:
        return False

    return cosine_distance(left_vector, right_vector) < distance_threshold


def pattern_similarity(
    left_edge: tuple[str, str, str],
    right_edge: tuple[str, str, str],
    entity_type_by_entity_id: Mapping[str, str],
    idf_weights: Mapping[str, float],
    type_idf_weights: Mapping[str, float],
    edge_type_embeddings: Mapping[str, list[float]],
    distance_threshold: float,
    entity_weight: float = 1.0,
    type_weight: float = 0.0,
) -> float:
    """
    Compare two (source_entity_id, target_entity_id, relationship_name)
    triples. The relationship is a hard gate, checked first: if it doesn't
    match (exactly, or close enough by embedding distance), the pair is
    discarded (score 0) without comparing source/target at all.

    If the relationship matches, source and target similarity are each
    computed with the same entity+type ``combined_similarity`` used at the
    summary level, just applied to single entities via singleton sets. The
    final score is the mean of the two endpoint similarities, keeping the
    result in [0, 1] like every other signal.
    """
    left_source, left_target, left_relationship_name = left_edge
    right_source, right_target, right_relationship_name = right_edge

    if not relationship_match(
        left_relationship_name, right_relationship_name, edge_type_embeddings, distance_threshold
    ):
        return 0.0

    def endpoint_similarity(left_entity_id: str, right_entity_id: str) -> float:
        entity_score = weighted_jaccard({left_entity_id}, {right_entity_id}, idf_weights)
        type_score = weighted_jaccard(
            type_ids_for_entities({left_entity_id}, entity_type_by_entity_id),
            type_ids_for_entities({right_entity_id}, entity_type_by_entity_id),
            type_idf_weights,
        )
        return combined_similarity(entity_score, type_score, 0.0, entity_weight, type_weight, 0.0)

    return (
        endpoint_similarity(left_source, right_source)
        + endpoint_similarity(left_target, right_target)
    ) / 2
