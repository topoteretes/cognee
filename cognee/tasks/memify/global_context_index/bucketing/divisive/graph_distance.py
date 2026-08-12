from __future__ import annotations

from collections.abc import Callable, Mapping

from ..graph.scoring import entities_weight, weighted_jaccard


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
) -> Callable[[str, str], float]:
    """
    Entity-overlap similarity between two items, reusing ``weighted_jaccard``
    exactly as the bottom-up strategy does. Only valid at level 0, where each
    item is a single TextSummary.
    """

    def similarity_fn(left_id: str, right_id: str) -> float:
        left_entities = entities_by_summary_id.get(left_id, set())
        right_entities = entities_by_summary_id.get(right_id, set())
        return weighted_jaccard(left_entities, right_entities, idf_weights)

    return similarity_fn


def build_graph_group_similarity_fn(
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
) -> Callable[[str, list[str]], float]:
    """
    Resolves a tie between the two poles: similarity of an item to the union
    of entities of a group's members, used only for items that scored
    exactly equal against both poles (see ``split.py``'s ``_resolve_ties``).
    """

    def group_similarity_fn(item_id: str, group_ids: list[str]) -> float:
        item_entities = entities_by_summary_id.get(item_id, set())
        group_entities: set[str] = set()
        for group_id in group_ids:
            group_entities.update(entities_by_summary_id.get(group_id, set()))
        return weighted_jaccard(item_entities, group_entities, idf_weights)

    return group_similarity_fn
