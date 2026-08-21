"""Merge two hybrid retrievals into one result of the same shape.

Each channel merges under its own budget, and ``chunk_summaries`` is rebuilt for the chunks
that survived so it never references a dropped chunk. Keys the retriever carries but this
module does not own are taken from the primary result unchanged.
"""

from typing import Any, Optional

from cognee.modules.retrieval.hybrid.results import empty_hybrid_result, result_id
from cognee.modules.retrieval.utils.merge_results import conversational_reserve, merge_ranked

_DERIVED_KEYS = frozenset(empty_hybrid_result())


def merge_hybrid_results(
    primary: Optional[dict],
    secondary: Optional[dict],
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
