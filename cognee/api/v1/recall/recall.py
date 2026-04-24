import re
from uuid import UUID
from typing import Optional, Union

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict

from cognee.memory.entries import normalize_scope
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
    retriever_specific_config: dict
    user: object


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens using word boundaries."""
    return {w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) >= _MIN_WORD_LEN}


async def _resolve_user_id(user) -> Optional[str]:
    """Return the user id as a string, resolving default if needed."""
    from cognee.modules.users.methods import get_default_user

    if user is None:
        user = await get_default_user()
    return str(user.id) if hasattr(user, "id") else None


async def _resolve_session_cache_user_id(
    session_id: str, caller_user_id: Optional[str]
) -> Optional[str]:
    """Resolve the user_id to use when querying the session cache.

    Session-cache entries are keyed by the session's OWNER, not by the
    authenticated caller. A caller may legitimately query someone
    else's session via a dataset read grant — in that case we need to
    return the owner's id so ``SessionManager.get_session`` finds the
    entries.

    Two complications the resolver handles:

    * The same ``session_id`` can exist under multiple owners (the PK
      is ``(session_id, user_id)``, not ``session_id`` alone). The
      caller might own an empty row AND simultaneously have read
      permission on a non-owned row that has all the cache content.
      We pick the candidate most likely to have real entries — the
      one with ``dataset_id`` populated wins over an empty row.

    * Falls back to ``caller_user_id`` when nothing is in
      ``session_records`` yet so behaviour matches the pre-visibility
      state.
    """
    if not session_id:
        return caller_user_id
    try:
        from uuid import UUID

        from sqlalchemy import select
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.session_lifecycle.models import SessionRecord
        from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
            get_specific_user_permission_datasets,
        )

        caller_uuid = UUID(caller_user_id) if caller_user_id else None
        if caller_uuid is None:
            return caller_user_id

        permitted_ids: list = []
        try:
            permitted = await get_specific_user_permission_datasets(caller_uuid, "read", None)
            permitted_ids = [ds.id for ds in permitted] if permitted else []
        except Exception:
            permitted_ids = []

        # Fetch ALL candidate rows the caller can see for this
        # session_id. Owner match OR permitted-dataset match.
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            stmt = select(SessionRecord).where(SessionRecord.session_id == session_id)
            rows = list((await session.execute(stmt)).scalars().all())

        if not rows:
            return caller_user_id

        visible: list[SessionRecord] = []
        for r in rows:
            if r.user_id == caller_uuid:
                visible.append(r)
            elif permitted_ids and r.dataset_id in permitted_ids:
                visible.append(r)

        if not visible:
            return caller_user_id

        # Prefer a row with dataset_id populated — those came through
        # the proper write path and have cache content. Within that,
        # prefer rows the caller does NOT own (the active writer of
        # the session, e.g. an agent).
        with_dataset = [r for r in visible if r.dataset_id is not None]
        pool = with_dataset or visible
        non_owner = [r for r in pool if r.user_id != caller_uuid]
        chosen = (non_owner or pool)[0]
        owner = getattr(chosen, "user_id", None)
        return str(owner) if owner is not None else caller_user_id
    except Exception:
        pass
    return caller_user_id


async def _search_session(
    query_text: str,
    session_id: str,
    top_k: int = 10,
    user=None,
    _parent_span=None,
) -> list:
    """Search session-cache QA entries by keyword matching.

    Tokenizes the query and each QA entry (question + context + answer),
    ranks by token overlap, returns the top_k tagged with
    ``_source: "session"``.
    """
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    caller_user_id = await _resolve_user_id(user)
    if not caller_user_id:
        return []
    # Cache is keyed by session owner — resolve cross-user grants.
    cache_user_id = await _resolve_session_cache_user_id(session_id, caller_user_id)
    if not cache_user_id:
        return []

    sm = get_session_manager()
    if not sm.is_available:
        return []

    entries = await sm.get_session(
        user_id=cache_user_id,
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

        entry_text = " ".join(
            str(entry.get(field, "")) for field in ("question", "context", "answer")
        )
        entry_words = _tokenize(entry_text)

        hits = len(query_words & entry_words)
        if hits > 0:
            scored.append((hits, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for _, entry in scored[:top_k]:
        tagged = dict(entry)
        tagged["_source"] = "session"
        results.append(tagged)

    return results


async def _search_trace(
    query_text: str,
    session_id: str,
    top_k: int = 10,
    user=None,
) -> list:
    """Search session-cache agent trace steps by keyword matching.

    Tokenizes over origin_function, serialized method_params,
    method_return_value, memory_query, memory_context, and
    session_feedback. Returns top_k tagged with ``_source: "trace"``.
    """
    import json

    from cognee.infrastructure.session.get_session_manager import get_session_manager

    caller_user_id = await _resolve_user_id(user)
    if not caller_user_id:
        return []
    cache_user_id = await _resolve_session_cache_user_id(session_id, caller_user_id)
    if not cache_user_id:
        return []

    sm = get_session_manager()
    if not sm.is_available:
        return []

    entries = await sm.get_agent_trace_session(user_id=cache_user_id, session_id=session_id)

    if not entries:
        return []

    query_words = _tokenize(query_text)
    if not query_words:
        return []

    scored = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        parts = [
            str(entry.get("origin_function", "")),
            str(entry.get("status", "")),
            str(entry.get("memory_query", "")),
            str(entry.get("memory_context", "")),
            str(entry.get("session_feedback", "")),
            str(entry.get("error_message", "")),
        ]
        mp = entry.get("method_params")
        if mp is not None:
            try:
                parts.append(json.dumps(mp, ensure_ascii=False))
            except Exception:
                parts.append(str(mp))
        mrv = entry.get("method_return_value")
        if mrv is not None:
            try:
                parts.append(json.dumps(mrv, ensure_ascii=False))
            except Exception:
                parts.append(str(mrv))

        entry_words = _tokenize(" ".join(parts))
        hits = len(query_words & entry_words)
        if hits > 0:
            scored.append((hits, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for _, entry in scored[:top_k]:
        tagged = dict(entry)
        tagged["_source"] = "trace"
        results.append(tagged)

    return results


async def _fetch_graph_context(session_id: str, user=None) -> list:
    """Return the graph-context snapshot for the session as a one-item list.

    ``improve()`` writes a distilled summary of graph knowledge into
    ``graph_knowledge:{user}:{session}`` — this surfaces it as a recall
    result tagged with ``_source: "graph_context"`` (or an empty list
    when nothing has been synced yet).
    """
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    caller_user_id = await _resolve_user_id(user)
    if not caller_user_id:
        return []
    cache_user_id = await _resolve_session_cache_user_id(session_id, caller_user_id)
    if not cache_user_id:
        return []

    sm = get_session_manager()
    if not sm.is_available:
        return []

    snapshot = await sm.get_graph_context(user_id=cache_user_id, session_id=session_id)
    if not snapshot:
        return []

    return [{"_source": "graph_context", "content": snapshot}]


async def recall(
    query_text: str,
    query_type: Optional[SearchType] = None,
    *,
    datasets: Optional[list[str]] = None,
    top_k: int = 10,
    auto_route: bool = True,
    scope: Optional[Union[str, list[str]]] = None,
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
    user = kwargs.get("user")

    # Resolve scope → concrete source list. "auto" (the default) picks
    # sources based on what the caller supplied:
    #
    # * session_id alone (no datasets, no query_type):
    #     session → graph, short-circuit on session hit (legacy behaviour).
    # * session_id + datasets and/or query_type:
    #     session AND graph, both contribute (legacy "auto" scope).
    # * no session_id:
    #     graph only.
    #
    # Explicit ``scope`` values bypass this entirely.
    resolved_scope = normalize_scope(scope)
    if resolved_scope == ["auto"]:
        if session_id and not datasets and query_type is None:
            sources = ["session", "graph"]
            auto_fallthrough = True  # session hit short-circuits graph
        elif session_id:
            sources = ["session", "graph"]
            auto_fallthrough = False  # both contribute
        else:
            sources = ["graph"]
            auto_fallthrough = False
    else:
        sources = resolved_scope
        auto_fallthrough = False

    span_scope = ",".join(sources)

    send_telemetry(
        "cognee.recall",
        kwargs.get("user", "sdk"),
        additional_properties={
            "query_length": len(query_text),
            "scope": span_scope,
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
        span.set_attribute(COGNEE_RECALL_SCOPE, span_scope)
        if session_id:
            span.set_attribute(COGNEE_SESSION_ID, session_id)
        span.set_attribute("cognee.recall.top_k", top_k)

        from cognee.api.v1.serve.state import get_remote_client

        client = get_remote_client()
        if client is not None:
            results = await client.recall(
                query_text,
                query_type,
                datasets=datasets,
                top_k=top_k,
                scope=scope,
                **kwargs,
            )
            span.set_attribute(COGNEE_RECALL_SOURCE, "cloud")
            span.set_attribute(COGNEE_RESULT_COUNT, len(results) if results else 0)
            return results

        merged: list = []

        async def _run_session():
            if not session_id:
                return []
            return await _search_session(
                query_text=query_text,
                session_id=session_id,
                top_k=top_k,
                user=user,
            )

        async def _run_trace():
            if not session_id:
                return []
            return await _search_trace(
                query_text=query_text,
                session_id=session_id,
                top_k=top_k,
                user=user,
            )

        async def _run_graph_context():
            if not session_id:
                return []
            return await _fetch_graph_context(session_id=session_id, user=user)

        async def _run_graph():
            from cognee.api.v1.search import search

            local_query_type = query_type
            if local_query_type is not None:
                if auto_route:
                    from cognee.api.v1.recall.query_router import route_query, record_override

                    result = route_query(query_text)
                    routed_type = result.search_type
                    record_override(routed_type, local_query_type)
            elif auto_route:
                from cognee.api.v1.recall.query_router import route_query

                result = route_query(query_text)
                local_query_type = result.search_type
            else:
                local_query_type = SearchType.GRAPH_COMPLETION

            span.set_attribute(
                COGNEE_SEARCH_TYPE,
                str(local_query_type.value) if local_query_type else "unknown",
            )

            graph_response = await search(
                query_text=query_text,
                query_type=local_query_type,
                datasets=datasets,
                top_k=top_k,
                **kwargs,
            )
            tagged: list = []
            for item in graph_response.results:
                row = item.model_dump(mode="json")
                row["_source"] = "graph"
                tagged.append(row)
            return tagged

        runners = {
            "session": _run_session,
            "trace": _run_trace,
            "graph_context": _run_graph_context,
            "graph": _run_graph,
        }

        session_result_count = 0
        for src in sources:
            runner = runners.get(src)
            if runner is None:
                continue
            # Auto mode special case: session hit short-circuits graph.
            if auto_fallthrough and src == "graph" and merged:
                break
            part = await runner()
            if src == "session":
                session_result_count = len(part)
            merged.extend(part)

        if session_result_count:
            span.set_attribute(COGNEE_SESSION_ENTRY_COUNT, session_result_count)

        # Choose a single-source label when only one source contributed,
        # else "multi".
        source_label = sources[0] if len(sources) == 1 else "multi"
        span.set_attribute(COGNEE_RECALL_SOURCE, source_label)
        span.set_attribute(COGNEE_RESULT_COUNT, len(merged))

        logger.info(
            "recall: %d results across sources=%s (session=%s)",
            len(merged),
            sources,
            session_id or "-",
        )

        return merged
