"""Merge two hybrid retrievals into one result of the same shape.

Each channel merges under its own budget, and ``chunk_summaries`` is rebuilt for the chunks
that survived so it never references a dropped chunk. Keys the retriever carries but this
module does not own are taken from the primary result unchanged.
"""

from collections.abc import Hashable
from typing import Any

from cognee.modules.retrieval.hybrid.entities import _edge_dedupe_key
from cognee.modules.retrieval.hybrid.results import empty_hybrid_result
from cognee.modules.retrieval.utils.merge_results import conversational_reserve, merge_ranked
from cognee.modules.retrieval.utils.results import result_id

_DERIVED_KEYS = frozenset(empty_hybrid_result())


def merge_hybrid_results(
    primary: dict | None,
    secondary: dict | None,
    *,
    chunks_limit: int,
    entities_limit: int,
    facts_limit: int,
) -> dict:
    """Merge each hybrid channel while preserving the result shape and its budgets."""
    primary = primary or {}
    secondary = secondary or {}
    entities = merge_ranked(
        primary.get("entities"),
        secondary.get("entities"),
        limit=entities_limit,
        secondary_reserve=conversational_reserve(entities_limit),
    )
    channels: dict[str, list] = {
        "chunks": merge_ranked(
            primary.get("chunks"),
            secondary.get("chunks"),
            limit=chunks_limit,
            secondary_reserve=conversational_reserve(chunks_limit),
        ),
        "entities": entities,
        "facts": merge_ranked(
            drop_facts_shown_as_edges(primary.get("facts") or [], entities),
            drop_facts_shown_as_edges(secondary.get("facts") or [], entities),
            limit=facts_limit,
            secondary_reserve=conversational_reserve(facts_limit),
        ),
    }

    merged: dict[str, Any] = {
        key: value for key, value in primary.items() if key not in _DERIVED_KEYS
    }
    merged.update(channels)

    chunk_ids = [chunk_id for chunk in channels["chunks"] if (chunk_id := result_id(chunk))]
    primary_summaries = primary.get("chunk_summaries", {})
    secondary_summaries = secondary.get("chunk_summaries", {})
    merged["chunk_summaries"] = {
        chunk_id: summary
        for chunk_id in chunk_ids
        if (summary := primary_summaries.get(chunk_id) or secondary_summaries.get(chunk_id))
    }
    return merged


def merge_hybrid_results_union(results: list[dict | None]) -> dict:
    """Union peer hybrid results into one, round-robin by rank, deduplicated by id.

    Every result's first item comes before any result's second; at equal rank the
    earlier result comes first. An item several results returned appears once, at
    its best rank. An entity keeps the edges every result showed for it, and a fact
    already shown as an entity edge is dropped, as within one hybrid retrieval.
    Items without an id are never merged away.
    """
    results = [result for result in results or [] if result]
    if not results:
        return empty_hybrid_result()

    channels: dict[str, list] = {
        channel: _union_ranked([result.get(channel) or [] for result in results])
        for channel in ("chunks", "entities", "facts")
    }
    channels["entities"] = [
        _with_all_edges(entity, results) if isinstance(entity, dict) else entity
        for entity in channels["entities"]
    ]
    channels["facts"] = drop_facts_shown_as_edges(channels["facts"], channels["entities"])

    merged: dict[str, Any] = {
        key: value for key, value in results[0].items() if key not in _DERIVED_KEYS
    }
    merged.update(channels)

    summaries = [result.get("chunk_summaries") or {} for result in results]
    merged["chunk_summaries"] = {
        chunk_id: summary
        for chunk in channels["chunks"]
        if (chunk_id := result_id(chunk))
        and (summary := next((s[chunk_id] for s in summaries if s.get(chunk_id)), None))
    }
    return merged


def drop_facts_shown_as_edges(facts: list, entities: list) -> list:
    """Facts minus those an entity already shows as an edge (``select_facts``' rule)."""
    bullet_ids = {
        edge["edge_type_id"]
        for entity in entities
        if isinstance(entity, dict)
        for edge in entity.get("edges", [])
        if edge.get("edge_type_id")
    }
    return [fact for fact in facts if result_id(fact) not in bullet_ids]


def _union_ranked(columns: list[list]) -> list:
    seen: set[Hashable] = set()
    merged: list = []
    for rank in range(max(map(len, columns), default=0)):
        for lane, column in enumerate(columns):
            if rank >= len(column):
                continue
            item = column[rank]
            key: Hashable = result_id(item) or ("unidentified", lane, rank)
            if key not in seen:
                seen.add(key)
                merged.append(item)
    return merged


def _with_all_edges(entity: dict, results: list[dict]) -> dict:
    """The entity with the edges every result showed for it, kept in first-seen order."""
    entity_id = result_id(entity)
    if not entity_id:
        return entity
    edges: list = []
    seen: set[Hashable] = set()
    for result in results:
        for candidate in result.get("entities") or []:
            if not isinstance(candidate, dict) or result_id(candidate) != entity_id:
                continue
            for edge in candidate.get("edges", []):
                key = _edge_dedupe_key(edge) or edge.get("text")
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)
    return {**entity, "edges": edges}
