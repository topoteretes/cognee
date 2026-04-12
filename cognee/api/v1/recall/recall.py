import re
from uuid import UUID
from typing import Optional

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict

from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger
from cognee.modules.observability import (
    new_span,
    COGNEE_SEARCH_QUERY,
    COGNEE_SEARCH_TYPE,
    COGNEE_SESSION_ID,
    COGNEE_RESULT_COUNT,
    COGNEE_RECALL_SCOPE,
    COGNEE_RECALL_SOURCE,
    COGNEE_SESSION_ENTRY_COUNT,
)

logger = get_logger("recall")

# Minimum word length to avoid matching noise words like "a", "I"
_MIN_WORD_LEN = 2


class RecallKwargs(TypedDict, total=False):
    """Power-user overrides for recall(). Most users never need these."""

    dataset_ids: list[UUID]
    system_prompt: str
    system_prompt_path: str
    node_name: list[str]
    node_name_filter_operator: str
    only_context: bool
    session_id: str
    wide_search_top_k: int
    triplet_distance_penalty: float
    feedback_influence: float
    verbose: bool
    retriever_specific_config: dict
    user: object


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens using word boundaries."""
    return {w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) >= _MIN_WORD_LEN}


async def _search_session(
    query_text: str,
    session_id: str,
    top_k: int = 10,
    user=None,
    _parent_span=None,
) -> list:
    """Search session cache entries by word-boundary keyword matching.

    Tokenizes both the query and each QA entry's fields, then ranks
    entries by the number of overlapping words. Returns results tagged
    with ``_source: "session"`` so callers can distinguish session
    results from graph results.
    """
    from cognee.infrastructure.session.get_session_manager import get_session_manager
    from cognee.modules.users.methods import get_default_user

    if user is None:
        user = await get_default_user()

    user_id = str(user.id) if hasattr(user, "id") else None
    if not user_id:
        return []

    sm = get_session_manager()
    if not sm.is_available:
        return []

    entries = await sm.get_session(
        user_id=user_id,
        session_id=session_id,
        formatted=False,
    )

    if not isinstance(entries, list) or not entries:
        return []

    query_words = _tokenize(query_text)
    if not query_words:
        return []

    scored = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        # Tokenize the searchable fields
        entry_text = " ".join(
            str(entry.get(field, "")) for field in ("question", "context", "answer")
        )
        entry_words = _tokenize(entry_text)

        hits = len(query_words & entry_words)
        if hits > 0:
            scored.append((hits, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Tag results with source
    results = []
    for _, entry in scored[:top_k]:
        tagged = dict(entry)
        tagged["_source"] = "session"
        results.append(tagged)

    return results


async def recall(
    query_text: str,
    query_type: Optional[SearchType] = None,
    *,
    datasets: Optional[list[str]] = None,
    top_k: int = 10,
    auto_route: bool = True,
    **kwargs: Unpack[RecallKwargs],
) -> list:
    """Search the knowledge graph for relevant information.

    When ``session_id`` is provided without ``datasets`` or
    ``query_type``, searches session cache entries directly by keyword
    matching. This returns matching QA entries from the session without
    hitting the permanent graph. If no session entries match, falls
    through to the permanent graph search.

    Each result dict includes a ``_source`` key (``"session"`` or
    ``"graph"``) so callers can tell where the result came from.

    When ``query_type`` is omitted and ``auto_route`` is True (default),
    a lightweight rule-based classifier picks the best search strategy.
    Set ``auto_route=False`` to skip the classifier and use
    GRAPH_COMPLETION as the default, or pass ``query_type`` explicitly.

    Args:
        query_text: Natural-language query.
        query_type: Search strategy. When provided, the router is bypassed.
        datasets: Dataset names to search within.
        top_k: Maximum results to return (default *10*).
        auto_route: If True and query_type is None, classify the query
            automatically. If False, fall back to GRAPH_COMPLETION.
        **kwargs: Additional options -- see ``RecallKwargs``.

    Returns:
        Search results. When searching session-only, returns a list of
        matching QA entry dicts with ``_source="session"``.
    """
    from cognee.shared.utils import send_telemetry
    from cognee import __version__ as cognee_version

    session_id = kwargs.get("session_id")
    scope = "session" if (session_id and not datasets and query_type is None) else "graph"
    if session_id and datasets:
        scope = "auto"

    send_telemetry(
        "cognee.recall",
        kwargs.get("user", "sdk"),
        additional_properties={
            "query_length": len(query_text),
            "scope": scope,
            "auto_route": auto_route,
            "top_k": top_k,
            "search_type": str(query_type.value) if query_type else "auto",
            "session_id": session_id or "",
            "datasets": ",".join(datasets) if datasets else "",
            "cognee_version": cognee_version,
        },
    )

    with new_span("cognee.api.recall") as span:
        span.set_attribute(COGNEE_SEARCH_QUERY, query_text[:500])
        span.set_attribute(COGNEE_RECALL_SCOPE, scope)
        if session_id:
            span.set_attribute(COGNEE_SESSION_ID, session_id)
        span.set_attribute("cognee.recall.top_k", top_k)

        from cognee.api.v1.serve.state import get_remote_client

        client = get_remote_client()
        if client is not None:
            results = await client.recall(
                query_text, query_type, datasets=datasets, top_k=top_k, **kwargs
            )
            span.set_attribute(COGNEE_RECALL_SOURCE, "cloud")
            span.set_attribute(COGNEE_RESULT_COUNT, len(results) if results else 0)
            return results

        # Session-only search: when session_id is provided but no datasets
        # or explicit query_type, search the session cache directly.
        if session_id and not datasets and query_type is None:
            user = kwargs.get("user")
            session_results = await _search_session(
                query_text=query_text,
                session_id=session_id,
                top_k=top_k,
                user=user,
            )
            if session_results:
                logger.info(
                    "recall: found %d session entries for session '%s'",
                    len(session_results),
                    session_id,
                )
                span.set_attribute(COGNEE_RECALL_SOURCE, "session")
                span.set_attribute(COGNEE_RESULT_COUNT, len(session_results))
                span.set_attribute(COGNEE_SESSION_ENTRY_COUNT, len(session_results))
                return session_results
            logger.info("recall: no session entries matched, falling through to graph search")

        from cognee.api.v1.search import search

        if query_type is not None:
            if auto_route:
                from cognee.api.v1.recall.query_router import route_query, record_override

                result = route_query(query_text)
                routed_type = result.search_type
                record_override(routed_type, query_type)
        elif auto_route:
            from cognee.api.v1.recall.query_router import route_query

            result = route_query(query_text)
            query_type = result.search_type
        else:
            query_type = SearchType.GRAPH_COMPLETION

        span.set_attribute(COGNEE_SEARCH_TYPE, str(query_type.value) if query_type else "unknown")

        graph_results = await search(
            query_text=query_text,
            query_type=query_type,
            datasets=datasets,
            top_k=top_k,
            **kwargs,
        )

        # Tag graph results with source
        tagged = []
        for r in graph_results:
            if isinstance(r, dict):
                r["_source"] = "graph"
                tagged.append(r)
            else:
                tagged.append(r)

        span.set_attribute(COGNEE_RECALL_SOURCE, "graph")
        span.set_attribute(COGNEE_RESULT_COUNT, len(tagged))
        return tagged
