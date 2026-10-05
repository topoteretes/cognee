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


def empty_anchors() -> dict:
    return {"timestamp_ids": set(), "chunk_ids": set(), "entity_ids": set()}


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
