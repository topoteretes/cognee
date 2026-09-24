"""Helpers for TemporalHybridRetriever: interval extraction, matching, filtering.

build_temporal_index derives query-independent lookup tables from one graph
snapshot; match_temporal_neighborhood answers one query window against them,
so the snapshot is walked once per run instead of once per query.
"""

from datetime import datetime, timezone

from temporal_extraction_task import timestamp_bounds

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.retrieval.hybrid.results import result_id
from cognee.shared.logging_utils import get_logger
from cognee.tasks.temporal_graph.models import QueryInterval

logger = get_logger("temporal_matching")

QUERY_INTERVAL_PROMPT = """Extract one time window from the question.
Return starts_at and ends_at as UTC calendar fields, or null when that side is open.

Start is inclusive and end is exclusive:
- A named year, month, or day covers that whole unit.
- An inclusive range includes the entire final named unit.
- A precise second covers that second; end is the next second.
- "Before X" leaves start null and sets end to X's lower bound.
- "After X" sets start to X's upper bound and leaves end null.
- No explicit time, relative time, or disjoint windows: both null.

Every non-null boundary must include year, month, day, hour, minute, and second.
Do not guess relative dates. Do not return more than one window.
"""


def _boundary_datetime(boundary) -> datetime | None:
    if boundary is None:
        return None
    return datetime(**boundary.model_dump(), tzinfo=timezone.utc)


async def extract_query_interval(
    query: str,
) -> tuple[datetime | None, datetime | None, str | None]:
    interval = await LLMGateway.acreate_structured_output(
        text_input=query,
        system_prompt=QUERY_INTERVAL_PROMPT,
        response_model=QueryInterval,
    )
    try:
        start = _boundary_datetime(interval.starts_at)
        end = _boundary_datetime(interval.ends_at)
    except (TypeError, ValueError):
        return None, None, "invalid_interval"
    if start is None and end is None:
        return None, None, "no_time_constraint"
    if start is not None and end is not None and start >= end:
        return None, None, "invalid_interval"
    return start, end, None


def _overlaps(
    lower: datetime, upper: datetime, start: datetime | None, end: datetime | None
) -> bool:
    return (end is None or lower < end) and (start is None or upper > start)


def build_temporal_index(nodes, edges) -> dict:
    """Derive the query-independent lookup tables from one graph snapshot."""
    nodes_by_id = {str(node_id): properties or {} for node_id, properties in nodes}
    contains_by_chunk = {}
    begins_at = {}
    ends_at = {}
    at_edges = []
    for source, target, relationship, _properties in edges:
        source_id, target_id, name = str(source), str(target), str(relationship)
        if name == "contains" and nodes_by_id.get(source_id, {}).get("type") == "DocumentChunk":
            contains_by_chunk.setdefault(source_id, set()).add(target_id)
        if name == "begins_at":
            begins_at.setdefault(source_id, []).append(target_id)
        if name == "ends_at":
            ends_at.setdefault(source_id, []).append(target_id)
        if name.endswith("_at"):
            at_edges.append((source_id, target_id, name))

    timestamp_ranges = {}
    timestamp_labels = {}
    for node_id, properties in nodes_by_id.items():
        if properties.get("type") != "Timestamp":
            continue
        try:
            normalized, lower, upper = timestamp_bounds(str(properties.get("timestamp_str") or ""))
        except ValueError:
            logger.warning(
                "Skipping malformed timestamp id=%s value=%s",
                node_id,
                properties.get("timestamp_str"),
            )
            continue
        timestamp_ranges[node_id] = (lower, upper)
        timestamp_labels[node_id] = normalized

    periods = {}
    for owner_id in set(begins_at) | set(ends_at):
        start_targets = begins_at.get(owner_id, [])
        end_targets = ends_at.get(owner_id, [])
        if len(start_targets) != 1 or len(end_targets) != 1 or start_targets[0] == end_targets[0]:
            logger.warning("Skipping incomplete or ambiguous period owner=%s", owner_id)
            continue
        start_id, end_id = start_targets[0], end_targets[0]
        if start_id not in timestamp_ranges or end_id not in timestamp_ranges:
            logger.warning("Skipping period without timestamp bounds owner=%s", owner_id)
            continue
        lower, upper = timestamp_ranges[start_id][0], timestamp_ranges[end_id][1]
        if not lower < upper:
            logger.warning("Skipping reversed period owner=%s", owner_id)
            continue
        periods[owner_id] = (lower, upper, start_id, end_id)

    chunk_entities = {
        chunk_id: {
            target_id
            for target_id in contained
            if nodes_by_id.get(target_id, {}).get("type") == "Entity"
        }
        for chunk_id, contained in contains_by_chunk.items()
    }
    return {
        "contains_by_chunk": contains_by_chunk,
        "chunk_entities": chunk_entities,
        "timestamp_ranges": timestamp_ranges,
        "timestamp_labels": timestamp_labels,
        "periods": periods,
        "at_edges": at_edges,
    }


def match_temporal_neighborhood(index: dict, start, end) -> dict:
    """Answer one query window against a prebuilt temporal index."""
    timestamp_ids = {
        node_id
        for node_id, (lower, upper) in index["timestamp_ranges"].items()
        if _overlaps(lower, upper, start, end)
    }
    matched_periods = {
        owner_id: (start_id, end_id)
        for owner_id, (lower, upper, start_id, end_id) in index["periods"].items()
        if _overlaps(lower, upper, start, end)
    }

    eligible_chunk_ids = set()
    for chunk_id, contained in index["contains_by_chunk"].items():
        has_timestamp = bool(contained & timestamp_ids)
        has_period = any(
            owner_id in contained and start_id in contained and end_id in contained
            for owner_id, (start_id, end_id) in matched_periods.items()
        )
        if has_timestamp or has_period:
            eligible_chunk_ids.add(chunk_id)

    temporal_edges = {
        (source_id, target_id, name)
        for source_id, target_id, name in index["at_edges"]
        if target_id in timestamp_ids
    }
    temporal_edges.update(
        (owner_id, endpoint_id, name)
        for owner_id, (start_id, end_id) in matched_periods.items()
        for endpoint_id, name in ((start_id, "begins_at"), (end_id, "ends_at"))
    )
    return {
        "eligible_chunk_ids": eligible_chunk_ids,
        "chunk_entities": index["chunk_entities"],
        "temporal_edges": temporal_edges,
        # all labels, not only matched ids: a matched period's endpoint can
        # itself lie outside the query window and still needs a readable name
        "timestamp_labels": index["timestamp_labels"],
        "timestamp_ids": timestamp_ids,
        "period_ids": set(matched_periods),
    }


def empty_matches() -> dict:
    return {
        "eligible_chunk_ids": set(),
        "chunk_entities": {},
        "temporal_edges": set(),
        "timestamp_labels": {},
        "timestamp_ids": set(),
        "period_ids": set(),
    }


def _summaries_for(summaries: dict, chunks: list) -> dict:
    """Kept for the hybrid result shape; the inherited renderer ignores summaries."""
    chunk_ids = {result_id(chunk) for chunk in chunks}
    return {key: value for key, value in (summaries or {}).items() if str(key) in chunk_ids}


def slice_hybrid(candidates: dict, top_k: int) -> dict:
    """The plain hybrid view of the candidate set: first top_k of every section."""
    chunks = list(candidates.get("chunks") or [])[:top_k]
    return {
        "chunks": chunks,
        "chunk_summaries": _summaries_for(candidates.get("chunk_summaries"), chunks),
        "entities": list(candidates.get("entities") or [])[:top_k],
        "facts": list(candidates.get("facts") or [])[:top_k],
    }


def _relabel_timestamp(edge: dict, labels: dict) -> dict:
    """Core Timestamp nodes carry no name, so bullets label them by node id;
    rewrite the target side from the node's timestamp_str instead."""
    label = labels.get(str(edge.get("target_id")))
    if label is None:
        return edge
    relabeled = {**edge, "target": label}
    source = edge.get("source") or edge.get("source_id")
    relationship = edge.get("relationship")
    if source and relationship:
        relabeled["text"] = f"{source} -- {relationship} -- {label}"
    return relabeled


def _restrict_entity(entity: dict, matches: dict) -> dict:
    """Copy an entity, dropping the description and every non-temporal edge bullet."""
    kept = [
        edge
        for edge in entity.get("edges") or []
        if isinstance(edge, dict)
        and (
            str(edge.get("source_id")),
            str(edge.get("target_id")),
            edge.get("relationship"),
        )
        in matches["temporal_edges"]
    ]
    return {
        **entity,
        "description": None,
        "edges": [_relabel_timestamp(edge, matches["timestamp_labels"]) for edge in kept],
    }


def filter_hybrid(candidates: dict, matches: dict, top_k: int) -> dict | None:
    """The temporal view of the candidate set; None when no candidate chunk is eligible."""
    chunks = [
        chunk
        for chunk in candidates.get("chunks") or []
        if result_id(chunk) in matches["eligible_chunk_ids"]
    ][:top_k]
    if not chunks:
        return None
    allowed_entities = {
        entity_id
        for chunk in chunks
        for entity_id in matches["chunk_entities"].get(result_id(chunk), set())
    }
    return {
        "chunks": chunks,
        "chunk_summaries": _summaries_for(candidates.get("chunk_summaries"), chunks),
        "entities": [
            _restrict_entity(entity, matches)
            for entity in candidates.get("entities") or []
            if isinstance(entity, dict) and entity.get("id") in allowed_entities
        ][:top_k],
        "facts": [],
    }
