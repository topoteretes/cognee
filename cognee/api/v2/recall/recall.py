from uuid import UUID
from typing import Optional

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict

from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall")


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


async def _search_session(
    query_text: str,
    session_id: str,
    top_k: int = 10,
    user=None,
) -> list:
    """Search session cache entries by keyword matching.

    Scans all QA entries in the session and returns those whose
    question, context, or answer contain any word from the query.
    Results are ranked by number of matching words.
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

    # Keyword matching: split query into lowercase words
    query_words = set(query_text.lower().split())
    if not query_words:
        return []

    scored = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # Build searchable text from all QA fields
        searchable = " ".join(
            str(entry.get(field, ""))
            for field in ("question", "context", "answer")
        ).lower()

        hits = sum(1 for w in query_words if w in searchable)
        if hits > 0:
            scored.append((hits, entry))

    # Sort by hit count descending, take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]


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
    hitting the permanent graph.

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
        Search results (same as ``cognee.search()``). When searching
        session-only, returns a list of matching QA entry dicts.
    """
    session_id = kwargs.get("session_id")

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
            return session_results
        # Fall through to graph search if no session results
        logger.info(
            "recall: no session entries matched, falling through to graph search"
        )

    from cognee.api.v1.search import search

    if query_type is not None:
        if auto_route:
            from cognee.api.v2.recall.query_router import route_query, record_override

            result = route_query(query_text)
            routed_type = result.search_type
            record_override(routed_type, query_type)
    elif auto_route:
        from cognee.api.v2.recall.query_router import route_query

        result = route_query(query_text)
        query_type = result.search_type
    else:
        query_type = SearchType.GRAPH_COMPLETION

    return await search(
        query_text=query_text,
        query_type=query_type,
        datasets=datasets,
        top_k=top_k,
        **kwargs,
    )
