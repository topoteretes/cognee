from typing import Any, List, Optional
from uuid import UUID

from .log_query import log_query
from .log_result import log_result


def _completion_of(payload: Any) -> Optional[str]:
    """The text a user saw, preferring the completion over raw context."""
    for attribute in ("completion", "context"):
        value = getattr(payload, attribute, None)
        if value:
            return value if isinstance(value, str) else str(value)
    return None


async def log_search_history(
    query_text: str,
    query_type: str,
    user_id: UUID,
    search_results: List[Any],
    session_id: Optional[str] = None,
) -> None:
    """Record a searched question and its answers, one row per dataset.

    A search fans out over every dataset it is given and produces one payload
    per dataset. Collapsing that into a single query row loses which dataset
    was searched, and collapsing the answers into one blob loses which dataset
    answered — so each payload gets its own query and result row.

    A payload that carries no dataset of its own — which is what happens with
    access control disabled — records ``None``.
    """
    payloads = [item[0] if isinstance(item, tuple) else item for item in search_results] or [None]

    for payload in payloads:
        dataset_id = getattr(payload, "dataset_id", None)
        query = await log_query(query_text, query_type, user_id, dataset_id, session_id)
        # Every query keeps a result row even when the search produced no text
        # — CHUNKS and SUMMARIES return objects, not completions — so history
        # stays one result per query, as it was before this fanned out.
        await log_result(query.id, _completion_of(payload) or "", user_id, dataset_id, session_id)
