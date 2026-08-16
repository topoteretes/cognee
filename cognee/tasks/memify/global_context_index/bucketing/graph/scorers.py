from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from .scoring import (
    combined_similarity as _combine_scores,
)
from .scoring import (
    pattern_similarity as _edge_pattern_similarity,
)
from .scoring import (
    type_ids_for_entities,
    weighted_jaccard,
)

Triple = tuple[str, str, str]


@dataclass(frozen=True)
class EntityGroupProfile:
    """
    Precomputed view of a summary's (or a bucket's) entity set, so the four
    scorers below never re-derive type ids or relation triples per comparison
    -- build once per summary/bucket, reuse across every scorer call.
    """

    entity_ids: frozenset[str]
    type_ids: frozenset[str] = field(default_factory=frozenset)
    relations: tuple[Triple, ...] = ()


def build_entity_group_profile(
    entity_ids: Iterable[str],
    entity_type_by_entity_id: Mapping[str, str],
    relation_index: Mapping[str, list[Triple]],
) -> EntityGroupProfile:
    entity_id_set = frozenset(entity_ids)
    return EntityGroupProfile(
        entity_ids=entity_id_set,
        type_ids=frozenset(type_ids_for_entities(entity_id_set, entity_type_by_entity_id)),
        relations=tuple(_relations_within(entity_id_set, relation_index)),
    )


def _relations_within(
    entity_ids: frozenset[str],
    relation_index: Mapping[str, list[Triple]],
) -> list[Triple]:
    seen: set[Triple] = set()
    result: list[Triple] = []
    for entity_id in entity_ids:
        for triple in relation_index.get(entity_id, []):
            if triple in seen:
                continue
            source_id, target_id, _ = triple
            if source_id in entity_ids and target_id in entity_ids:
                seen.add(triple)
                result.append(triple)
    return result


def build_pattern_index(
    entity_relations: Iterable[Triple],
) -> dict[str, list[Triple]]:
    """Dataset-wide entity id -> triples mentioning it (as either endpoint)."""
    relation_index: dict[str, list[Triple]] = {}
    for triple in entity_relations:
        source_id, target_id, _ = triple
        relation_index.setdefault(source_id, []).append(triple)
        relation_index.setdefault(target_id, []).append(triple)
    return relation_index


Scorer = Callable[[EntityGroupProfile, EntityGroupProfile], float]


def build_entity_jaccard(idf_weights: Mapping[str, float]) -> Scorer:
    def entity_jaccard(summary: EntityGroupProfile, bucket: EntityGroupProfile) -> float:
        return weighted_jaccard(summary.entity_ids, bucket.entity_ids, idf_weights)

    return entity_jaccard


def build_type_jaccard(type_idf_weights: Mapping[str, float]) -> Scorer:
    def type_jaccard(summary: EntityGroupProfile, bucket: EntityGroupProfile) -> float:
        return weighted_jaccard(summary.type_ids, bucket.type_ids, type_idf_weights)

    return type_jaccard


def build_pattern_similarity(
    entity_type_by_entity_id: Mapping[str, str],
    idf_weights: Mapping[str, float],
    type_idf_weights: Mapping[str, float],
    edge_type_embeddings: Mapping[str, list[float]],
    pattern_distance_threshold: float,
    entity_weight: float,
    type_weight: float,
) -> Scorer:
    def pattern_similarity(summary: EntityGroupProfile, bucket: EntityGroupProfile) -> float:
        if not summary.relations or not bucket.relations:
            return 0.0

        return max(
            (
                _edge_pattern_similarity(
                    left_edge,
                    right_edge,
                    entity_type_by_entity_id,
                    idf_weights,
                    type_idf_weights,
                    edge_type_embeddings,
                    pattern_distance_threshold,
                    entity_weight,
                    type_weight,
                )
                for left_edge in summary.relations
                for right_edge in bucket.relations
            ),
            default=0.0,
        )

    return pattern_similarity


def build_combined_similarity(
    entity_jaccard: Scorer,
    type_jaccard: Scorer,
    pattern_similarity: Scorer,
    entity_weight: float,
    type_weight: float,
    pattern_weight: float,
) -> Scorer:
    def combined_similarity(summary: EntityGroupProfile, bucket: EntityGroupProfile) -> float:
        entity_score = entity_jaccard(summary, bucket)
        type_score = type_jaccard(summary, bucket) if type_weight > 0 else 0.0
        pattern_score = pattern_similarity(summary, bucket) if pattern_weight > 0 else 0.0
        return _combine_scores(
            entity_score, type_score, pattern_score, entity_weight, type_weight, pattern_weight
        )

    return combined_similarity


def build_scorers(
    idf_weights: Mapping[str, float],
    entity_type_by_entity_id: Mapping[str, str],
    type_idf_weights: Mapping[str, float],
    edge_type_embeddings: Mapping[str, list[float]],
    pattern_distance_threshold: float,
    entity_weight: float,
    type_weight: float,
    pattern_weight: float,
) -> dict[str, Scorer]:
    """
    Build the four summary-vs-bucket scorers, all sharing the uniform
    ``(summary: EntityGroupProfile, bucket: EntityGroupProfile) -> float``
    signature. "combined" is what real placement code calls; "entity"/"type"/
    "pattern" are exposed individually for direct testing/inspection.
    """
    entity_jaccard = build_entity_jaccard(idf_weights)
    type_jaccard = build_type_jaccard(type_idf_weights)
    pattern_similarity = build_pattern_similarity(
        entity_type_by_entity_id,
        idf_weights,
        type_idf_weights,
        edge_type_embeddings,
        pattern_distance_threshold,
        entity_weight,
        type_weight,
    )
    combined_similarity = build_combined_similarity(
        entity_jaccard, type_jaccard, pattern_similarity, entity_weight, type_weight, pattern_weight
    )

    return {
        "entity": entity_jaccard,
        "type": type_jaccard,
        "pattern": pattern_similarity,
        "combined": combined_similarity,
    }
