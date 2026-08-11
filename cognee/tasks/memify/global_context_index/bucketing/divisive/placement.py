from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .graph_distance import build_graph_pole_a_fn, build_graph_similarity_fn
from .split import divisive_split
from .vector_distance import (
    build_vector_pole_a_fn,
    build_vector_similarity_fn,
    embed_items_for_divisive_split,
)
from ..common import create_bucket_node, mark_bucket_for_persistence, record_bucket_assignment
from ..graph.placement import _partition_summaries
from ...bucketing_strategy import BucketingStrategyName
from ...models import BucketAssignment, SummaryNode

Triple = tuple[str, str, str]


async def build_divisive_buckets_for_level(
    items: list[SummaryNode],
    level: int,
    dataset_id: str,
    max_bucket_size: int,
    bucketing_strategy: BucketingStrategyName,
    vector_engine: Any,
    entities_by_summary_id: Mapping[str, set[str]],
    idf_weights: Mapping[str, float],
    entity_type_by_entity_id: Mapping[str, str] | None = None,
    type_idf_weights: Mapping[str, float] | None = None,
    entity_weight: float = 1.0,
    type_weight: float = 0.0,
    pattern_weight: float = 0.0,
    entity_relations: list[Triple] | None = None,
    edge_type_embeddings: Mapping[str, list[float]] | None = None,
    pattern_distance_threshold: float = 0.5,
) -> tuple[dict[str, SummaryNode], list[BucketAssignment]]:
    """
    Build one level of buckets top-down (divisive) for a dataset's very first,
    complete build. Only called when there are no existing buckets at this
    level (see build.py's ``is_first_build``/dispatch gating) -- incremental
    updates always fall through to the existing bottom-up placement code,
    which accepts these buckets unmodified because they have the same shape.

    Uses the entity/type/pattern graph signal only at level 0 with
    ``bucketing_strategy == "graph"``; every other case (level 0 with
    ``"vector"``, or any level >= 1 regardless of ``bucketing_strategy``) uses
    a vector-embedding signal instead. Pattern/entity signals are never used
    above level 0: a level >= 1 bucket's entities are a union across many
    original summaries, and relation triples would only need both endpoints
    inside that union to look "related" even if they were never observed
    together in any single summary.
    """
    if max_bucket_size < 1:
        raise ValueError("max_bucket_size must be at least 1.")
    if not items:
        return {}, []

    buckets_to_persist: dict[str, SummaryNode] = {}
    assignments: list[BucketAssignment] = []

    if level == 0 and bucketing_strategy == "graph":
        entity_items, misc_items = _partition_summaries(items, entities_by_summary_id, idf_weights)
        if entity_items:
            pole_a_fn = build_graph_pole_a_fn(entities_by_summary_id, idf_weights)
            similarity_fn = build_graph_similarity_fn(
                entities_by_summary_id,
                idf_weights,
                entity_type_by_entity_id,
                type_idf_weights,
                entity_weight,
                type_weight,
                entity_relations,
                edge_type_embeddings,
                pattern_weight,
                pattern_distance_threshold,
            )
            item_id_groups = divisive_split(
                [item.id for item in entity_items], similarity_fn, pole_a_fn, max_bucket_size
            )
            for child_ids in item_id_groups:
                _add_bucket(
                    child_ids,
                    dataset_id,
                    level,
                    buckets_to_persist,
                    assignments,
                    graph_bucket_entity_ids=_union_entities(child_ids, entities_by_summary_id),
                )

        for chunk in _chunk_ids(sorted(item.id for item in misc_items), max_bucket_size):
            _add_bucket(
                chunk,
                dataset_id,
                level,
                buckets_to_persist,
                assignments,
                graph_bucket_entity_ids=set(),
            )

        return buckets_to_persist, assignments

    vectors_by_id = await embed_items_for_divisive_split(items, vector_engine)
    pole_a_fn = build_vector_pole_a_fn(vectors_by_id)
    similarity_fn = build_vector_similarity_fn(vectors_by_id)
    item_id_groups = divisive_split(
        [item.id for item in items], similarity_fn, pole_a_fn, max_bucket_size
    )
    for child_ids in item_id_groups:
        _add_bucket(child_ids, dataset_id, level, buckets_to_persist, assignments)

    return buckets_to_persist, assignments


def _add_bucket(
    child_ids: list[str],
    dataset_id: str,
    level: int,
    buckets_to_persist: dict[str, SummaryNode],
    assignments: list[BucketAssignment],
    graph_bucket_entity_ids: set[str] | None = None,
) -> None:
    bucket = create_bucket_node(
        child_ids, dataset_id, level, graph_bucket_entity_ids=graph_bucket_entity_ids
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


def _chunk_ids(item_ids: list[str], chunk_size: int) -> list[list[str]]:
    return [item_ids[index : index + chunk_size] for index in range(0, len(item_ids), chunk_size)]
