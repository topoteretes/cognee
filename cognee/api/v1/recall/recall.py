import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from cognee.infrastructure.databases.cache import SessionAgentTraceEntry, SessionQAEntry
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from cognee.context_global_variables import set_session_user_context_variable
from cognee.exceptions import CogneeValidationError
from cognee.memory.entries import normalize_scope
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.observability import (
    COGNEE_RECALL_SCOPE,
    COGNEE_RECALL_SOURCE,
    COGNEE_RESULT_COUNT,
    COGNEE_SEARCH_QUERY,
    COGNEE_SEARCH_TYPE,
    COGNEE_SESSION_ENTRY_COUNT,
    COGNEE_SESSION_ID,
    new_span,
)
from cognee.modules.recall.types.RecallResponse import (
    RecallResponse,
    ResponseAgentTraceEntry,
    ResponseGraphContextEntry,
    ResponseGraphEntry,
    ResponseQAEntry,
)
from cognee.modules.recall.types.SearchResultItem import SearchResultItem
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchResult, SearchType
from cognee.modules.users.exceptions.exceptions import UserNotFoundError
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall")

# Minimum word length to avoid matching noise words like "a", "I"
_MIN_WORD_LEN = 2


class RecallKwargs(TypedDict, total=False):
    """Backward-compatible export for callers that import RecallKwargs."""

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


async def _resolve_user_id(user: str | None) -> str | None:
    """Return the user id as a string, resolving default if needed."""
    if user is None:
        user = await get_default_user()
    return str(user.id) if hasattr(user, "id") else None


async def _resolve_session_cache_user_id(session_id: str, caller_user_id: str | None) -> str | None:
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

        permitted_ids: list[UUID] = []
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
    user: str | None = None,
    _parent_span=None,
) -> list[ResponseQAEntry]:
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

    scored: list[tuple[int, SessionQAEntry]] = []
    for entry in entries:
        entry_text = " ".join((entry.question, entry.context, entry.answer))
        entry_words = _tokenize(entry_text)

        hits = len(query_words & entry_words)
        if hits > 0:
            scored.append((hits, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[ResponseQAEntry] = []
    for _, entry in scored[:top_k]:
        results.append(ResponseQAEntry(**entry.model_dump(), source="session"))

    return results


async def _search_trace(
    query_text: str,
    session_id: str,
    top_k: int = 10,
    user: str | None = None,
) -> list[ResponseAgentTraceEntry]:
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

    scored: list[tuple[int, SessionAgentTraceEntry]] = []
    for entry in entries:
        parts = [
            entry.origin_function,
            entry.status,
            entry.memory_query,
            entry.memory_context,
            entry.session_feedback,
            entry.error_message,
        ]
        mp = entry.method_params
        try:
            parts.append(json.dumps(mp, ensure_ascii=False))
        except Exception:
            parts.append(str(mp))
        mrv = entry.method_return_value
        try:
            parts.append(json.dumps(mrv, ensure_ascii=False))
        except Exception:
            parts.append(str(mrv))

        entry_words = _tokenize(" ".join(parts))
        hits = len(query_words & entry_words)
        if hits > 0:
            scored.append((hits, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[ResponseAgentTraceEntry] = []
    for _, entry in scored[:top_k]:
        results.append(ResponseAgentTraceEntry(**entry.model_dump(), source="trace"))

    return results


async def _fetch_graph_context(
    session_id: str, user: str | None = None
) -> list[ResponseGraphContextEntry]:
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

    return [ResponseGraphContextEntry(content=snapshot, source="graph_context")]


async def recall(
    query_text: str,
    query_type: SearchType | None = None,
    *,
    datasets: list[str] | None = None,
    dataset_ids: list[UUID] | None = None,
    top_k: int = 10,
    auto_route: bool = True,
    scope: str | list[str] | None = None,
    system_prompt: str | None = None,
    system_prompt_path: str = "answer_simple_question.txt",
    node_name: list[str] | None = None,
    node_name_filter_operator: str = "OR",
    only_context: bool = False,
    session_id: str | None = None,
    wide_search_top_k: int | None = 100,
    triplet_distance_penalty: float | None = 6.5,
    feedback_influence: float = 0.0,
    verbose: bool = False,
    retriever_specific_config: dict | None = None,
    neighborhood_depth: int | None = None,
    neighborhood_seed_top_k: int | None = None,
    user: object | None = None,
) -> list[RecallResponse]:
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
        dataset_ids: Dataset UUIDs to search within. Takes precedence over datasets.
        top_k: Maximum results to return (default *10*).
        auto_route: If True and query_type is None, classify the query
            automatically. If False, fall back to GRAPH_COMPLETION.

    Returns:
        Search results. When searching session-only, returns a list of
        matching QA entry dicts with ``_source="session"``.
    """
    from cognee import __version__ as cognee_version
    from cognee.shared.utils import send_telemetry

    telemetry_user = getattr(user, "id", user) or "sdk"

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
        has_dataset_scope = bool(dataset_ids) or bool(datasets)
        if session_id and not has_dataset_scope and query_type is None:
            sources = ["session", "graph"]
            auto_fallthrough = True  # session hit short-circuits graph
        elif session_id and query_type is None:
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
        telemetry_user,
        additional_properties={
            "query_length": len(query_text),
            "scope": span_scope,
            "auto_route": auto_route,
            "top_k": top_k,
            "search_type": str(query_type.value) if query_type else "auto",
            "session_id": session_id or "",
            "datasets": ",".join(datasets) if datasets else "",
            "dataset_ids": ",".join(str(dataset_id) for dataset_id in dataset_ids or []),
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
                dataset_ids=dataset_ids,
                top_k=top_k,
                scope=scope,
                system_prompt=system_prompt,
                node_name=node_name,
                only_context=only_context,
                session_id=session_id,
                verbose=verbose,
            )
            span.set_attribute(COGNEE_RECALL_SOURCE, "cloud")
            span.set_attribute(COGNEE_RESULT_COUNT, len(results) if results else 0)
            return results

        merged: list[RecallResponse] = []

        async def _run_session() -> list[RecallResponse]:
            if not session_id:
                return []
            return list(
                await _search_session(
                    query_text=query_text,
                    session_id=session_id,
                    top_k=top_k,
                    user=user,
                )
            )

        async def _run_trace() -> list[RecallResponse]:
            if not session_id:
                return []
            return list(
                await _search_trace(
                    query_text=query_text,
                    session_id=session_id,
                    top_k=top_k,
                    user=user,
                )
            )

        async def _run_graph_context() -> list[RecallResponse]:
            if not session_id:
                return []
            return list(await _fetch_graph_context(session_id=session_id, user=user))

        async def _run_graph() -> list[RecallResponse]:
            nonlocal user

            from cognee.modules.recall.methods.normalize_search_payload import (
                normalize_search_payload,
            )
            from cognee.modules.search.methods.search import authorized_search

            if user is None:
                try:
                    user = await get_default_user()
                except (DatabaseNotCreatedError, UserNotFoundError) as error:
                    raise CogneeValidationError(
                        message=(
                            "Recall prerequisites not met: no database/default user found. "
                            "Initialize Cognee before recalling by:\n"
                            "- running `await cognee.add(...)` followed by `await cognee.cognify()`."
                        ),
                        name="RecallPreconditionError",
                    ) from error

            await set_session_user_context_variable(user)

            local_query_type = query_type
            if local_query_type is not None:
                if auto_route:
                    from cognee.api.v1.recall.query_router import record_override, route_query

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

            # Dataset UUIDs take precedence over names, matching /api/v1/search.
            # String dataset names can only resolve for the current user.
            search_dataset_ids = dataset_ids or None
            if search_dataset_ids is None and datasets is not None:
                search_dataset_ids = [
                    dataset.id
                    for dataset in await get_authorized_existing_datasets(datasets, "read", user)
                ]
                if not search_dataset_ids:
                    raise DatasetNotFoundError(message="No datasets found.")

            graph_results = await authorized_search(
                query_text=query_text,
                query_type=local_query_type,
                user=user,
                dataset_ids=search_dataset_ids,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                top_k=top_k,
                node_name=node_name,
                node_name_filter_operator=node_name_filter_operator,
                only_context=only_context,
                session_id=session_id,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
                feedback_influence=feedback_influence,
                retriever_specific_config=retriever_specific_config,
                neighborhood_depth=neighborhood_depth,
                neighborhood_seed_top_k=neighborhood_seed_top_k,
            )

            tagged = []
            for r in graph_results:
                items: list[SearchResultItem] = normalize_search_payload(r)
                tagged.extend(
                    [ResponseGraphEntry(**item.model_dump(), source="graph") for item in items]
                )
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
