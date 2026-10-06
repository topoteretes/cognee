"""Helpers for TemporalHybridRetriever: the question's time window (SDK-828).

The retriever fetches an oversized hybrid candidate set, asks the graph adapter
which of those candidates are attached to a Timestamp inside the window
(``GraphDBInterface.get_temporal_anchors``), moves the anchored ones to the
front (``HybridCandidates.prioritize``) and cuts with the same ``finalize``
plain hybrid uses, so the temporal result is never smaller than plain hybrid.
"""

from datetime import datetime, timezone

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.retrieval.temporal_hybrid.models import QueryInterval

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
