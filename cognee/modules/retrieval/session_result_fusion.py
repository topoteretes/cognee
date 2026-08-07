from collections.abc import Callable, Hashable
from typing import Any

from cognee.modules.retrieval.hybrid.results import display_value, result_id

RRF_OFFSET = 60
RAW_WEIGHT = 2.0
CONTEXTUAL_WEIGHT = 1.0


def graph_identity(edge: Any) -> Hashable:
    """Return an edge object ID or a stable graph relationship identity."""
    if isinstance(edge, dict):
        attributes = edge.get("attributes") or edge.get("edge_attributes") or edge
        source = edge.get("source_id") or edge.get("source_node_id")
        target = edge.get("target_id") or edge.get("target_node_id")
        directed = edge.get("directed", True)
    else:
        attributes = getattr(edge, "attributes", {}) or {}
        source = getattr(getattr(edge, "node1", None), "id", None)
        target = getattr(getattr(edge, "node2", None), "id", None)
        directed = getattr(edge, "directed", True)

    if not isinstance(attributes, dict):
        attributes = {}

    edge_object_id = display_value(attributes.get("edge_object_id"))
    if edge_object_id:
        return ("edge", edge_object_id)

    relationship = display_value(
        attributes.get("relationship_name") or attributes.get("relationship")
    )
    return (
        "graph",
        display_value(source) or "",
        relationship or "",
        display_value(target) or "",
        bool(directed),
    )


def fuse_ranked_results(
    raw_results: list[Any] | None,
    contextual_results: list[Any] | None,
    *,
    identity: Callable[[Any], Hashable | None],
    limit: int | None,
) -> list[Any]:
    """Fuse two ranked lists with weighted RRF and deterministic tie-breaking."""
    records: dict[Hashable, dict[str, Any]] = {}

    for lane, results, weight in (
        ("raw", raw_results or [], RAW_WEIGHT),
        ("contextual", contextual_results or [], CONTEXTUAL_WEIGHT),
    ):
        for rank, item in enumerate(results, start=1):
            item_identity = identity(item)
            key = item_identity if item_identity is not None else ("missing", lane, rank)
            record = records.setdefault(
                key,
                {
                    "score": 0.0,
                    "raw_rank": float("inf"),
                    "contextual_rank": float("inf"),
                    "raw": None,
                    "contextual": None,
                },
            )
            rank_key = f"{lane}_rank"
            if record[rank_key] != float("inf"):
                continue
            record[rank_key] = rank
            record[lane] = item
            record["score"] += weight / (RRF_OFFSET + rank)

    ranked = sorted(
        records.items(),
        key=lambda item: (
            -item[1]["score"],
            item[1]["raw_rank"],
            item[1]["contextual_rank"],
            repr(item[0]),
        ),
    )
    if limit is not None:
        ranked = ranked[: max(0, limit)]
    return [
        record["raw"] if record["raw"] is not None else record["contextual"] for _, record in ranked
    ]


def fuse_vector_results(
    raw_results: list[Any] | None,
    contextual_results: list[Any] | None,
    *,
    limit: int,
) -> list[Any]:
    return fuse_ranked_results(
        raw_results,
        contextual_results,
        identity=result_id,
        limit=limit,
    )


def fuse_graph_results(
    raw_results: list[Any] | None,
    contextual_results: list[Any] | None,
    *,
    limit: int,
) -> list[Any]:
    return fuse_ranked_results(
        raw_results,
        contextual_results,
        identity=graph_identity,
        limit=limit,
    )


def merge_channel_status(
    raw_status: dict | None,
    contextual_status: dict | None,
    *,
    item_count: int,
) -> dict:
    """Merge two Hybrid channel statuses without hiding a successful lane."""
    statuses = [status for status in (raw_status, contextual_status) if isinstance(status, dict)]
    if any(status.get("status") == "ok" for status in statuses):
        return {"status": "ok", "item_count": item_count}
    if any(status.get("status") == "degraded" for status in statuses):
        failed = next(status for status in statuses if status.get("status") == "degraded")
        return dict(failed)
    if statuses:
        return dict(statuses[0])
    return {"status": "skipped"}


def _status(result: dict | None, channel: str) -> dict | None:
    statuses = result.get("retrieval_status") if isinstance(result, dict) else None
    status = statuses.get(channel) if isinstance(statuses, dict) else None
    return status if isinstance(status, dict) else None


def _chunk_attribution(result: dict | None) -> dict[str, dict]:
    if not isinstance(result, dict):
        return {}
    return {
        str(item["chunk_id"]): item
        for item in result.get("chunk_attribution", [])
        if isinstance(item, dict) and item.get("chunk_id") is not None
    }


def fuse_hybrid_results(
    raw_result: dict | None,
    contextual_result: dict | None,
    *,
    chunks_limit: int,
    entities_limit: int,
    facts_limit: int,
    graph_limit: int,
) -> dict:
    """Fuse each Hybrid channel while retaining its existing result shape."""
    raw_result = raw_result or {}
    contextual_result = contextual_result or {}
    channels = {
        "chunks": fuse_vector_results(
            raw_result.get("chunks"), contextual_result.get("chunks"), limit=chunks_limit
        ),
        "entities": fuse_vector_results(
            raw_result.get("entities"), contextual_result.get("entities"), limit=entities_limit
        ),
        "facts": fuse_vector_results(
            raw_result.get("facts"), contextual_result.get("facts"), limit=facts_limit
        ),
    }
    if "graph_fallback" in raw_result or "graph_fallback" in contextual_result:
        channels["graph_fallback"] = fuse_graph_results(
            raw_result.get("graph_fallback"),
            contextual_result.get("graph_fallback"),
            limit=graph_limit,
        )

    fused = {
        key: value
        for key, value in raw_result.items()
        if key
        not in {
            "chunks",
            "chunk_summaries",
            "chunk_attribution",
            "entities",
            "facts",
            "graph_fallback",
            "retrieval_status",
        }
    }
    fused.update(channels)

    raw_summaries = raw_result.get("chunk_summaries", {})
    contextual_summaries = contextual_result.get("chunk_summaries", {})
    fused["chunk_summaries"] = {
        chunk_id: raw_summaries.get(chunk_id) or contextual_summaries.get(chunk_id)
        for chunk in channels["chunks"]
        if (chunk_id := result_id(chunk))
        and (raw_summaries.get(chunk_id) or contextual_summaries.get(chunk_id))
    }

    raw_attribution = _chunk_attribution(raw_result)
    contextual_attribution = _chunk_attribution(contextual_result)
    attribution = [
        raw_attribution.get(chunk_id) or contextual_attribution.get(chunk_id)
        for chunk in channels["chunks"]
        if (chunk_id := result_id(chunk))
        and (raw_attribution.get(chunk_id) or contextual_attribution.get(chunk_id))
    ]
    if attribution:
        fused["chunk_attribution"] = attribution

    fused_status = {
        channel: merge_channel_status(
            _status(raw_result, channel),
            _status(contextual_result, channel),
            item_count=len(items),
        )
        for channel, items in channels.items()
    }
    global_status = _status(raw_result, "global_context") or _status(
        contextual_result, "global_context"
    )
    if global_status is not None:
        fused_status["global_context"] = dict(global_status)
    fused["retrieval_status"] = fused_status
    return fused
