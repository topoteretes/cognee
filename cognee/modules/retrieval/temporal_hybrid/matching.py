"""Helpers for TemporalHybridRetriever: interval extraction, anchors, reranking (SDK-828).

The retriever asks the graph adapter for the Timestamp nodes inside the question's
window (``GraphDBInterface.get_timestamps_in_range``), reads their one-hop
neighbourhood, and reorders the hybrid candidates so the ones anchored to those
timestamps come first. Nothing is dropped: a candidate with no temporal anchor
keeps its place behind the anchored ones, so the result is never smaller than
plain hybrid retrieval.
"""

from datetime import datetime, timezone

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.retrieval.hybrid.results import result_id
from cognee.tasks.temporal_graph.models import QueryInterval

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
    """``(start, end, reason)``: the question's window, or a reason there is none."""
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


def to_epoch_ms(moment: datetime | None) -> int | None:
    return None if moment is None else int(moment.timestamp() * 1000)


def anchors_from_neighborhood(timestamp_ids: set[str], nodes, edges) -> dict:
    """What the matched timestamps are attached to, read off their one-hop neighbourhood.

    ``nodes``/``edges`` are in ``get_graph_data`` shape. A chunk is anchored
    when it ``contains`` a matched timestamp; an entity is anchored when any of
    its edges points at one (``born_at``, ``occurred_on``, ``begins_at`` …: the
    relationship name is not inspected, the target is what matters).
    ``chunk_times`` / ``entity_times`` keep the matched timestamps' strings per
    anchored node, and ``entity_names`` the anchored entities' names, for the
    notes the context carries.
    """
    properties_by_id = {str(node_id): (properties or {}) for node_id, properties in nodes}
    types = {node_id: properties.get("type") for node_id, properties in properties_by_id.items()}
    chunk_ids: set[str] = set()
    entity_ids: set[str] = set()
    chunk_times: dict[str, set[str]] = {}
    entity_times: dict[str, set[str]] = {}
    for source, target, relationship, _properties in edges:
        source_id, target_id = str(source), str(target)
        if target_id not in timestamp_ids:
            continue
        time_name = str(properties_by_id.get(target_id, {}).get("timestamp_str") or target_id)
        if types.get(source_id) == "DocumentChunk" and relationship == "contains":
            chunk_ids.add(source_id)
            chunk_times.setdefault(source_id, set()).add(time_name)
        elif types.get(source_id) == "Entity":
            entity_ids.add(source_id)
            entity_times.setdefault(source_id, set()).add(time_name)
    entity_names = {
        entity_id: str(properties_by_id.get(entity_id, {}).get("name") or entity_id)
        for entity_id in entity_ids
    }
    return {
        "chunk_ids": chunk_ids,
        "entity_ids": entity_ids,
        "chunk_times": chunk_times,
        "entity_times": entity_times,
        "entity_names": entity_names,
        "chunk_via": {},
    }


def chunks_containing(entity_ids: set[str], nodes, edges) -> dict[str, set[str]]:
    """Chunk id -> the ids in ``entity_ids`` its ``contains`` edges point at."""
    types = {str(node_id): (properties or {}).get("type") for node_id, properties in nodes}
    via: dict[str, set[str]] = {}
    for source, target, relationship, _properties in edges:
        source_id, target_id = str(source), str(target)
        if (
            relationship == "contains"
            and target_id in entity_ids
            and types.get(source_id) == "DocumentChunk"
        ):
            via.setdefault(source_id, set()).add(target_id)
    return via


def empty_anchors() -> dict:
    return {
        "timestamp_ids": set(),
        "chunk_ids": set(),
        "entity_ids": set(),
        "chunk_times": {},
        "entity_times": {},
        "entity_names": {},
        "chunk_via": {},
    }


UNDATED_NOTE = "time: not dated inside the window"


def passage_notes(chunks: list, anchors: dict) -> dict[str, str]:
    """One ``time:`` line per passage: the matched dates it contains, the dates
    it inherits through an anchored entity it mentions, or a statement that it
    has neither — so the model never has to guess which passages the window
    actually matched."""
    notes: dict[str, str] = {}
    for chunk in chunks:
        chunk_id = result_id(chunk)
        if chunk_id is None:
            continue
        own = anchors["chunk_times"].get(chunk_id)
        if own:
            notes[chunk_id] = "time: " + ", ".join(sorted(own))
            continue
        via = anchors["chunk_via"].get(chunk_id)
        if via:
            times = sorted(
                {t for entity_id in via for t in anchors["entity_times"].get(entity_id, ())}
            )
            names = sorted(anchors["entity_names"].get(entity_id, entity_id) for entity_id in via)
            notes[chunk_id] = f"time: {', '.join(times)} (through {', '.join(names)})"
            continue
        notes[chunk_id] = UNDATED_NOTE
    return notes


def _bound(moment: datetime) -> str:
    if (moment.hour, moment.minute, moment.second) == (0, 0, 0):
        return moment.strftime("%Y-%m-%d")
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def window_preamble(start: datetime | None, end: datetime | None) -> str:
    """The section that states the question's window and how to read the notes."""
    if start is not None and end is not None:
        period = f"{_bound(start)} to {_bound(end)}"
    elif end is not None:
        period = f"before {_bound(end)}"
    else:
        period = f"from {_bound(start)} onward"
    lines = [
        "## Time window",
        f"Question period: {period} (UTC, end exclusive).",
        (
            'A passage whose "time:" line names a date is dated inside this period in the '
            f'graph. A passage marked "{UNDATED_NOTE}" has no such date: judge its timing '
            "from its own text, and do not assume it refers to the question's period."
        ),
    ]
    return "\n".join(lines)


def _summaries_for(summaries: dict, chunks: list) -> dict:
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


def _anchored_first(items: list, anchored: set[str]) -> list:
    """Stable partition: anchored items in their original order, then the rest."""
    first = [item for item in items if result_id(item) in anchored]
    rest = [item for item in items if result_id(item) not in anchored]
    return first + rest


def rerank_hybrid(candidates: dict, anchors: dict, top_k: int) -> dict:
    """The temporal view of the candidate set: anchored candidates first, then the rest.

    Chunks and entities are reordered by a stable partition on membership in
    ``anchors`` and cut to ``top_k``; facts keep the plain hybrid order.
    Candidate contents are never altered — an entity keeps its description and
    every edge bullet — so a question whose window matches nothing relevant
    degrades to the plain hybrid slice rather than to an empty context.
    """
    chunks = _anchored_first(list(candidates.get("chunks") or []), anchors["chunk_ids"])[:top_k]
    entities = _anchored_first(
        [entity for entity in candidates.get("entities") or [] if isinstance(entity, dict)],
        anchors["entity_ids"],
    )[:top_k]
    return {
        "chunks": chunks,
        "chunk_summaries": _summaries_for(candidates.get("chunk_summaries"), chunks),
        "entities": entities,
        "facts": list(candidates.get("facts") or [])[:top_k],
    }
