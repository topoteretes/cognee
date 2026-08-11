from __future__ import annotations

from collections.abc import Callable, Mapping

from ..graph.scoring import (
    combined_similarity,
    entities_weight,
    pattern_similarity,
    type_ids_for_entities,
    weighted_jaccard,
)

Triple = tuple[str, str, str]


def build_graph_pole_a_fn(
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
) -> Callable[[list[str]], str]:
    """
    Pole A: the item with the highest aggregate entity distinctiveness (same
    signal ``_sort_entity_seeds`` uses in the bottom-up strategy). Ties break
    on ascending id.
    """

    def pole_a_fn(ids: list[str]) -> str:
        return sorted(
            ids,
            key=lambda item_id: (
                -entities_weight(entities_by_summary_id.get(item_id, set()), idf_weights),
                item_id,
            ),
        )[0]

    return pole_a_fn


def build_graph_similarity_fn(
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
    entity_type_by_entity_id: Mapping[str, str] | None = None,
    type_idf_weights: Mapping[str, float] | None = None,
    entity_weight: float = 1.0,
    type_weight: float = 0.0,
    entity_relations: list[Triple] | None = None,
    edge_type_embeddings: Mapping[str, list[float]] | None = None,
    pattern_weight: float = 0.0,
    pattern_distance_threshold: float = 0.5,
) -> Callable[[str, str], float]:
    """
    Entity + entity-type + relationship-pattern similarity between two items,
    reusing ``combined_similarity`` exactly as COG-6129 built it. Only valid
    at level 0, where each item is a single TextSummary: relation triples are
    looked up per individual item's own entity set, never a union across
    multiple summaries (see ``_relations_by_item`` below).
    """
    entity_type_by_entity_id = entity_type_by_entity_id or {}
    type_idf_weights = type_idf_weights or {}
    edge_type_embeddings = edge_type_embeddings or {}
    relations_by_item = _relations_by_item(entities_by_summary_id, entity_relations or [])

    def similarity_fn(left_id: str, right_id: str) -> float:
        left_entities = entities_by_summary_id.get(left_id, set())
        right_entities = entities_by_summary_id.get(right_id, set())

        entity_score = weighted_jaccard(left_entities, right_entities, idf_weights)
        type_score = weighted_jaccard(
            type_ids_for_entities(left_entities, entity_type_by_entity_id),
            type_ids_for_entities(right_entities, entity_type_by_entity_id),
            type_idf_weights,
        )
        pattern_score = 0.0
        if pattern_weight > 0:
            left_relations = relations_by_item.get(left_id, [])
            right_relations = relations_by_item.get(right_id, [])
            if left_relations and right_relations:
                pattern_score = max(
                    (
                        pattern_similarity(
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
                        for left_edge in left_relations
                        for right_edge in right_relations
                    ),
                    default=0.0,
                )

        return combined_similarity(
            entity_score, type_score, pattern_score, entity_weight, type_weight, pattern_weight
        )

    return similarity_fn


def _relations_by_item(
    entities_by_summary_id: Mapping[str, set[str]],
    entity_relations: list[Triple],
) -> dict[str, list[Triple]]:
    relation_index: dict[str, list[Triple]] = {}
    for triple in entity_relations:
        source_id, target_id, _ = triple
        relation_index.setdefault(source_id, []).append(triple)
        relation_index.setdefault(target_id, []).append(triple)

    relations_by_item: dict[str, list[Triple]] = {}
    for item_id, entity_ids in entities_by_summary_id.items():
        seen: set[Triple] = set()
        relevant: list[Triple] = []
        for entity_id in entity_ids:
            for triple in relation_index.get(entity_id, []):
                if triple in seen:
                    continue
                source_id, target_id, _ = triple
                if source_id in entity_ids and target_id in entity_ids:
                    seen.add(triple)
                    relevant.append(triple)
        if relevant:
            relations_by_item[item_id] = relevant

    return relations_by_item
