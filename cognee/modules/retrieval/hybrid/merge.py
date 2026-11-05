"""Merge two hybrid retrievals into one result of the same shape.

Each channel merges under its own budget, and ``chunk_summaries`` is rebuilt for the chunks
that survived so it never references a dropped chunk. Keys the retriever carries but this
module does not own are taken from the primary result unchanged.
"""

from collections.abc import Callable, Hashable
from typing import Any

from cognee.modules.retrieval.hybrid.results import empty_hybrid_result
from cognee.modules.retrieval.utils.merge_results import (
    conversational_reserve,
    edge_identity,
    merge_ranked,
)
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
    channels: dict[str, list] = {
        "chunks": merge_ranked(
            primary.get("chunks"),
            secondary.get("chunks"),
            limit=chunks_limit,
            secondary_reserve=conversational_reserve(chunks_limit),
        ),
        "entities": merge_ranked(
            primary.get("entities"),
            secondary.get("entities"),
            limit=entities_limit,
            secondary_reserve=conversational_reserve(entities_limit),
        ),
        "facts": merge_ranked(
            primary.get("facts"),
            secondary.get("facts"),
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


def merge_hybrid_results_union(
    results: list[dict | None],
    *,
    chunks_limit: int | None,
    entities_limit: int | None,
    facts_limit: int | None,
    preserve_first: bool = False,
) -> dict:
    """Union peer hybrid results into one, content-keyed, round-robin by rank.

    Legs are peers: every leg's first item comes before any leg's second, the
    first leg winning ties, so a channel that hits its limit still represents
    every leg. An item several legs returned appears once, at the position of
    its best rank and as the object from the leg that ranked it there (the
    earliest such leg on ties). Results without an identity are never merged
    away. With ``preserve_first=True``, every distinct item from the first
    result is retained using its original payload, even if a later leg ranks
    the same identity higher. Limits cap additional coverage, with that baseline
    as a minimum, even when a caller supplies a smaller ceiling.
    """
    results = [result for result in results or [] if result]
    if not results:
        return empty_hybrid_result()

    channels: dict[str, list] = {
        "chunks": _union_ranked(
            [result.get("chunks") for result in results],
            chunks_limit,
            preserve_first=preserve_first,
        ),
        "entities": _union_ranked(
            [result.get("entities") for result in results],
            entities_limit,
            preserve_first=preserve_first,
        ),
        "facts": _union_ranked(
            [result.get("facts") for result in results],
            facts_limit,
            identity=_fact_identity,
            preserve_first=preserve_first,
        ),
    }

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


def _fact_identity(item: Any) -> Hashable | None:
    identifier = result_id(item)
    if identifier is not None:
        return identifier
    key = edge_identity(item)
    # A graph edge needs a real identity. Do not collapse arbitrary anonymous
    # fact text into the all-empty graph tuple.
    if key[0] == "edge" or all(key[1:4]):
        return key
    return None


def _union_ranked(
    lanes: list[list | None],
    limit: int | None,
    *,
    identity: Callable[[Any], Hashable | None] = result_id,
    preserve_first: bool = False,
) -> list:
    columns = [list(lane or []) for lane in lanes]
    cap = None if limit is None else max(0, limit)
    if cap == 0 and not preserve_first:
        return []
    baseline_items = {}
    if preserve_first:
        for rank, item in enumerate(columns[0] if columns else []):
            key = identity(item)
            if key is None:
                key = ("unidentified", 0, rank)
            baseline_items.setdefault(key, item)
    baseline = set(baseline_items)
    if cap is not None:
        cap = max(cap, len(baseline))
    seen: set[Hashable] = set()
    merged: list = []
    for rank in range(max((len(column) for column in columns), default=0)):
        for lane, column in enumerate(columns):
            if rank >= len(column):
                continue
            item = column[rank]
            key: Hashable = identity(item) or ("unidentified", lane, rank)
            if key in seen:
                continue
            seen.add(key)
            if cap is not None and key not in baseline and len(merged) + len(baseline) >= cap:
                continue
            baseline.discard(key)
            # A later leg may rank the same ID earlier, but must not replace
            # the baseline object's source metadata, edges or provenance.
            merged.append(baseline_items.get(key, item))
            if cap is not None and len(merged) >= cap:
                return merged
    return merged
