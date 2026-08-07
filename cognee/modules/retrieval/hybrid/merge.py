"""Merge two hybrid retrievals into one result of the same shape.

Each channel merges on its own terms, and everything derived from the chunk channel —
summaries, attribution, per-channel status — is rebuilt for the chunks that survived.
``global_context`` is deliberately not merged: it is built once from the raw query.
"""

from typing import Any, Optional

from cognee.modules.retrieval.hybrid.results import result_id
from cognee.modules.retrieval.utils.merge_results import edge_identity, merge_ranked


def merge_channel_status(
    primary_status: Optional[dict],
    secondary_status: Optional[dict],
    *,
    item_count: int,
) -> dict:
    """Report a channel as succeeded when either retrieval succeeded."""
    statuses = [status for status in (primary_status, secondary_status) if isinstance(status, dict)]
    if any(status.get("status") == "ok" for status in statuses):
        return {"status": "ok", "item_count": item_count}
    if any(status.get("status") == "degraded" for status in statuses):
        return dict(next(status for status in statuses if status.get("status") == "degraded"))
    if statuses:
        return dict(statuses[0])
    return {"status": "skipped"}


def _channel_status(result: Optional[dict], channel: str) -> Optional[dict]:
    statuses = result.get("retrieval_status") if isinstance(result, dict) else None
    status = statuses.get(channel) if isinstance(statuses, dict) else None
    return status if isinstance(status, dict) else None


def _chunk_attribution(result: Optional[dict]) -> dict[str, dict]:
    if not isinstance(result, dict):
        return {}
    return {
        str(item["chunk_id"]): item
        for item in result.get("chunk_attribution", [])
        if isinstance(item, dict) and item.get("chunk_id") is not None
    }


# Rebuilt from the merged channels rather than carried over from the primary result.
_DERIVED_KEYS = {
    "chunks",
    "chunk_summaries",
    "chunk_attribution",
    "entities",
    "facts",
    "graph_fallback",
    "retrieval_status",
}


def merge_hybrid_results(
    primary: Optional[dict],
    secondary: Optional[dict],
    *,
    chunks_limit: int,
    entities_limit: int,
    facts_limit: int,
    graph_limit: int,
) -> dict:
    """Merge each hybrid channel while preserving the result shape and its budgets."""
    primary = primary or {}
    secondary = secondary or {}
    channels: dict[str, list] = {
        "chunks": merge_ranked(primary.get("chunks"), secondary.get("chunks"), limit=chunks_limit),
        "entities": merge_ranked(
            primary.get("entities"), secondary.get("entities"), limit=entities_limit
        ),
        "facts": merge_ranked(primary.get("facts"), secondary.get("facts"), limit=facts_limit),
    }
    if "graph_fallback" in primary or "graph_fallback" in secondary:
        channels["graph_fallback"] = merge_ranked(
            primary.get("graph_fallback"),
            secondary.get("graph_fallback"),
            identity=edge_identity,
            limit=graph_limit,
        )

    merged = {key: value for key, value in primary.items() if key not in _DERIVED_KEYS}
    merged.update(channels)

    chunk_ids = [chunk_id for chunk in channels["chunks"] if (chunk_id := result_id(chunk))]

    primary_summaries = primary.get("chunk_summaries", {})
    secondary_summaries = secondary.get("chunk_summaries", {})
    merged["chunk_summaries"] = {
        chunk_id: summary
        for chunk_id in chunk_ids
        if (summary := primary_summaries.get(chunk_id) or secondary_summaries.get(chunk_id))
    }

    primary_attribution = _chunk_attribution(primary)
    secondary_attribution = _chunk_attribution(secondary)
    attribution = [
        entry
        for chunk_id in chunk_ids
        if (entry := primary_attribution.get(chunk_id) or secondary_attribution.get(chunk_id))
    ]
    if attribution:
        merged["chunk_attribution"] = attribution

    merged_status = {
        channel: merge_channel_status(
            _channel_status(primary, channel),
            _channel_status(secondary, channel),
            item_count=len(items),
        )
        for channel, items in channels.items()
    }
    global_status = _channel_status(primary, "global_context") or _channel_status(
        secondary, "global_context"
    )
    if global_status is not None:
        merged_status["global_context"] = dict(global_status)
    merged["retrieval_status"] = merged_status
    return merged
