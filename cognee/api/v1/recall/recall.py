import asyncio
import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from cognee.base_config import get_base_config
from cognee.context_global_variables import set_session_user_context_variable
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.databases.cache import SessionAgentTraceEntry, SessionQAEntry
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from cognee.memory.entries import normalize_scope
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.operations import get_current_operation, record_operation
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
    ResponseCodeEntry,
    ResponseGraphEntry,
    ResponseMarkerEntry,
    ResponseQAEntry,
    ResponseSessionContextEntry,
    ResponseSkillEntry,
    ResponseToolEntry,
)
from cognee.modules.recall.types.SearchResultItem import SearchResultItem
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import ContextFormat, SearchResult, SearchType
from cognee.modules.users.exceptions.exceptions import UserNotFoundError
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall")

# Minimum word length to avoid matching noise words like "a", "I"
_MIN_WORD_LEN = 2


class RecallKwargs(TypedDict, total=False):
    """Backward-compatible export for callers that import RecallKwargs."""

    system_prompt: str
    system_prompt_path: str
    node_name: list[str]
    node_name_filter_operator: str
    only_context: bool
    context_format: str
    session_id: str
    wide_search_top_k: int
    triplet_distance_penalty: float
    feedback_influence: float
    verbose: bool
    retriever_specific_config: dict
    response_model: type
    user: object


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens using word boundaries."""
    return {w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) >= _MIN_WORD_LEN}


async def _resolve_user_id(user: str | None) -> str | None:
    """Return the user id as a string, resolving default if needed."""
    if user is None:
        user = await get_default_user()
    current_operation = get_current_operation()
    if current_operation is not None:
        current_operation.set_user(user)
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
        from cognee.modules.users.permissions.methods import get_permitted_dataset_ids

        caller_uuid = UUID(caller_user_id) if caller_user_id else None
        if caller_uuid is None:
            return caller_user_id

        permitted_ids = await get_permitted_dataset_ids(caller_uuid)

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
    top_k: int = 15,
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
    top_k: int = 15,
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


async def _fetch_session_context(
    query_text: str,
    session_id: str,
    context_profile: str,
    user: str | None = None,
) -> list[ResponseSessionContextEntry]:
    """Render active session-context lessons for one profile as a one-item list.

    Read-only: uses the deterministic builder with ``stamp_served=False`` so it never updates
    served metadata. Returns an empty list when no lessons match the profile.
    """
    from cognee.infrastructure.session.get_session_manager import get_session_manager
    from cognee.infrastructure.session.session_context_builder import build_active_context_block

    caller_user_id = await _resolve_user_id(user)
    if not caller_user_id:
        return []
    cache_user_id = await _resolve_session_cache_user_id(session_id, caller_user_id)
    if not cache_user_id:
        return []

    sm = get_session_manager()
    if not sm.is_available:
        return []

    block, _served = await build_active_context_block(
        session_manager=sm,
        user_id=cache_user_id,
        session_id=session_id,
        query=query_text,
        context_profile=context_profile,
        stamp_served=False,
    )
    if not block:
        return []

    return [
        ResponseSessionContextEntry(
            content=block, context_profile=context_profile, source="session_context"
        )
    ]


def _scope_should_forward_resolved(scope: str | list[str] | None) -> bool:
    if isinstance(scope, str):
        return scope in {"all", "graph_context"}
    return bool(scope and {"all", "graph_context"}.intersection(scope))


async def recall(
    query_text: str,
    query_type: SearchType | None = None,
    *,
    datasets: list[str] | None = None,
    dataset_ids: list[UUID] | None = None,
    top_k: int = 15,
    auto_route: bool = True,
    scope: str | list[str] | None = None,
    system_prompt: str | None = None,
    system_prompt_path: str = "answer_simple_question.txt",
    node_name: list[str] | None = None,
    node_name_filter_operator: str = "OR",
    # only_context / verbose inspect retriever-specific shapes. Pin query_type:
    # unspecified hybrid may defer to GRAPH_COMPLETION, and search history
    # still records the type recall chose, not the deferred one.
    only_context: bool = False,
    context_format: ContextFormat | str = ContextFormat.CONTEXT,
    session_id: str | None = None,
    context_profile: str = "qa",
    wide_search_top_k: int | None = None,
    triplet_distance_penalty: float | None = None,
    feedback_influence: float = get_base_config().default_feedback_influence,
    verbose: bool = False,
    retriever_specific_config: dict | None = None,
    response_model: type | None = None,
    neighborhood_depth: int | None = None,
    neighborhood_seed_top_k: int | None = None,
    include_references: bool = False,
    tool_connections: list[str] | None = None,
    tools_trigger: str = "always",
    code_query: dict | None = None,
    user: object | None = None,
    llm_config: LLMConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
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
    HYBRID_COMPLETION as the default, or pass ``query_type`` explicitly.

    Args:
        query_text: Natural-language query.
        query_type: Search strategy. When provided, the router is bypassed.
        datasets: Dataset names to search within.
        dataset_ids: Dataset UUIDs to search within. Takes precedence over datasets.
        top_k: Maximum results to return (default *15*).
        auto_route: If True and query_type is None, classify the query
            automatically. If False, fall back to GRAPH_COMPLETION.
        response_model: Pydantic model class for structured completion output.
            Forwarded to the retriever, which validates the LLM answer against
            it; each result then carries the validated payload as a dict in its
            ``structured`` field. Shorthand for
            ``retriever_specific_config={"response_model": ...}`` — pass it in
            one place only. Supported by the completion-style search types
            (GRAPH_COMPLETION and variants, RAG/TRIPLET/HYBRID/TEMPORAL/
            AGENTIC completion). With a remote Cognee server the model's JSON
            Schema is sent (``model_json_schema()``); the server validates
            structure only — Python-side validators do not travel.
        tool_connections: Names of authorized external database connections
            for the ``"tools"`` scope. ``None`` uses every connection visible
            to the user. Only consulted when ``scope`` includes ``"tools"``
            (which is never implied by ``"auto"`` or ``"all"``) and
            ``TOOL_CALLS_ENABLED=true``.
        tools_trigger: When to run the ``"tools"`` scope: ``"always"``
            (default) or ``"on_empty"`` — go back to the original data source
            only when every other requested source returned nothing, i.e. when
            cognee lacks the context to answer.
        code_query: Structured operation and arguments for the ``"code"``
            scope (same dict format as ``search(code_query=...)``, e.g.
            ``{"operation": "impact_analysis", "seeds": ["UserService"]}``).
            ``None`` runs the default ``explore`` operation with the query
            text as seed. Only valid when ``scope`` includes ``"code"``
            (which is never implied by ``"auto"`` or ``"all"``); results are
            tagged ``_source="code"``. A seed the code graph cannot resolve
            contributes nothing rather than failing the recall.

    Returns:
        Search results. When searching session-only, returns a list of
        matching QA entry dicts with ``_source="session"``.
    """
    from cognee import __version__ as cognee_version
    from cognee.shared.utils import send_telemetry

    # Fold the first-class response_model param into retriever_specific_config,
    # the channel the retriever registry already reads. Doing this up front means
    # every downstream path (graph search, scope routing) sees one merged config.
    if response_model is not None:
        configured_model = (retriever_specific_config or {}).get("response_model")
        if configured_model is not None and configured_model is not response_model:
            raise CogneeValidationError(
                message="response_model was passed both directly and in "
                "retriever_specific_config with different values; pass it once."
            )
        retriever_specific_config = {
            **(retriever_specific_config or {}),
            "response_model": response_model,
        }

    # Pass the User through rather than pre-resolving its id: send_telemetry
    # reads both id and tenant_id off it.
    telemetry_user = user or "sdk"

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

    if tools_trigger not in ("always", "on_empty"):
        raise CogneeValidationError(
            message=f"Invalid tools_trigger '{tools_trigger}'. Valid values: 'always', 'on_empty'.",
            name="InvalidToolsTriggerError",
        )
    context_format = ContextFormat.parse(context_format)
    if code_query is not None and "code" not in sources:
        raise CogneeValidationError(
            message=(
                "code_query requires the 'code' scope — pass scope=['code'] or "
                "scope=['graph', 'code'] ('code' is never implied by 'auto' or 'all')."
            ),
            name="InvalidCodeQueryError",
        )
    # "on_empty" means: go back to the source database only when cognee lacks
    # context — so tools must observe every other source's results first.
    if tools_trigger == "on_empty" and "tools" in sources:
        sources = [source for source in sources if source != "tools"] + ["tools"]

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
            "include_references": include_references,
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

        forward_scope = sources if _scope_should_forward_resolved(scope) else scope

        client = get_remote_client()
        if client is not None:
            # A model class cannot cross the HTTP boundary — send its JSON
            # Schema instead; the server rebuilds a validation model from it
            # and results come back with the validated dict in `structured`.
            # Read from the merged config so the dict form travels too.
            remote_response_model = (retriever_specific_config or {}).get("response_model")
            results = await client.recall(
                query_text,
                query_type,
                datasets=datasets,
                dataset_ids=dataset_ids,
                top_k=top_k,
                scope=forward_scope,
                system_prompt=system_prompt,
                node_name=node_name,
                only_context=only_context,
                context_format=context_format,
                session_id=session_id,
                context_profile=context_profile,
                verbose=verbose,
                include_references=include_references,
                response_schema=(
                    remote_response_model.model_json_schema()
                    if remote_response_model is not None
                    else None
                ),
                tool_connections=tool_connections,
                tools_trigger=tools_trigger,
                code_query=code_query,
            )
            span.set_attribute(COGNEE_RECALL_SOURCE, "cloud")
            span.set_attribute(COGNEE_RESULT_COUNT, len(results) if results else 0)
            return results

        async with record_operation("recall", user=user, session_id=session_id):
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

            async def _run_session_context() -> list[RecallResponse]:
                if not session_id:
                    return []
                return list(
                    await _fetch_session_context(
                        query_text=query_text,
                        session_id=session_id,
                        context_profile=context_profile,
                        user=user,
                    )
                )

            async def _run_graph() -> list[RecallResponse]:
                nonlocal user, dataset_ids

                from cognee.modules.recall.methods.normalize_search_payload import (
                    normalize_search_payload,
                )

                from cognee.modules.search.methods.search import authorized_search
                from cognee.modules.search.operations import log_search_history

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

                current_operation = get_current_operation()
                if current_operation is not None:
                    current_operation.set_user(user)

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
                    local_query_type = SearchType.HYBRID_COMPLETION

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
                        for dataset in await get_authorized_existing_datasets(
                            datasets, "read", user
                        )
                    ]
                    if not search_dataset_ids:
                        raise DatasetNotFoundError(message="No datasets found.")

                from cognee.modules.recall.config import get_recall_config

                # Warm-up short-circuit. Config errors fail open (skip the
                # guard) so a malformed RECALL_WARMUP_* value can never take
                # recall down; only_context callers skip it too because they
                # expect context, not a marker.
                recall_config = None
                try:
                    recall_config = get_recall_config()
                except Exception as error:
                    logger.warning(
                        "Recall warm-up config failed to load; skipping guard: %s", error
                    )
                guard_active = (
                    recall_config is not None
                    and recall_config.recall_warmup_shortcircuit
                    and not only_context
                )
                probe_dataset_ids = search_dataset_ids
                if guard_active and dataset_ids:
                    from cognee.modules.users.exceptions import PermissionDeniedError
                    from cognee.modules.users.permissions.methods import (
                        get_specific_user_permission_datasets,
                    )

                    # Authorize caller-supplied dataset ids *before* probing —
                    # the same check authorized_search performs — so the guard
                    # can neither leak other tenants' processing state nor
                    # mask the PermissionDeniedError authorized_search would
                    # raise for unpermitted or nonexistent ids. Infrastructure
                    # errors fail open (skip the guard): authorized_search
                    # then performs the authoritative check as before.
                    try:
                        probe_dataset_ids = [
                            dataset.id
                            for dataset in await get_specific_user_permission_datasets(
                                user.id, "read", dataset_ids
                            )
                        ]
                    except PermissionDeniedError:
                        raise
                    except Exception as error:
                        logger.warning(
                            "Recall warm-up pre-probe authorization failed; skipping guard: %s",
                            error,
                        )
                        guard_active = False

                if guard_active:
                    from cognee.modules.recall.methods.graph_warmup import (
                        STATE_BUILD_FAILED,
                        assess_memory_readiness,
                    )

                    probe = await assess_memory_readiness(user, probe_dataset_ids)
                    if not probe.is_warm:
                        logger.info(
                            "Recall warm-up short-circuit: graph readiness is '%s' "
                            "(threshold %d); skipping search.",
                            probe.state,
                            recall_config.recall_warmup_threshold,
                        )
                        span.set_attribute("cognee.recall.warmup_shortcircuit", True)
                        span.set_attribute("cognee.recall.warmup_state", probe.state)
                        if sources != ["graph"]:
                            # Multi-source recall: a cold graph contributes
                            # nothing, so other lanes — and the tools
                            # "on_empty" fallback, which fires only when the
                            # merged result is empty — behave exactly as if
                            # the graph lane returned no results.
                            return []
                        if probe.state == STATE_BUILD_FAILED:
                            failure_desc = probe.error_class or "unknown error"
                            if probe.error_message:
                                failure_desc = f"{failure_desc}: {probe.error_message}"
                            status = "build_failed"
                            text = (
                                "Memory build failed: the last ingestion for the requested "
                                f"datasets ended in an error ({failure_desc}). Fix the cause "
                                "and re-run remember() or cognify()."
                            )
                        else:
                            status = "memory_warming_up"
                            text = (
                                "Memory is still warming up: no knowledge graph data "
                                "exists yet for the requested datasets."
                            )
                        return [
                            ResponseMarkerEntry(
                                source="system",
                                status=status,
                                text=text,
                                datapoint_count=probe.datapoint_count,
                                threshold=recall_config.recall_warmup_threshold,
                                error_class=probe.error_class,
                                error_message=probe.error_message,
                            )
                        ]

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
                    context_format=context_format,
                    session_id=session_id,
                    wide_search_top_k=wide_search_top_k,
                    triplet_distance_penalty=triplet_distance_penalty,
                    feedback_influence=feedback_influence,
                    retriever_specific_config=retriever_specific_config,
                    neighborhood_depth=neighborhood_depth,
                    neighborhood_seed_top_k=neighborhood_seed_top_k,
                    include_references=include_references,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                )

                # /v1/search records every question it answers; recall never did,
                # because it calls authorized_search() directly and skips the
                # logging that search() wraps around it. Agents recall through this
                # endpoint, so their questions were absent from history entirely.
                await log_search_history(query_text, local_query_type.value, user.id, graph_results)

                tagged = []
                for r in graph_results:
                    items: list[SearchResultItem] = normalize_search_payload(r)
                    tagged.extend(
                        [ResponseGraphEntry(**item.model_dump(), source="graph") for item in items]
                    )
                return tagged

            async def _run_tools() -> list[RecallResponse]:
                from uuid import UUID as _UUID

                from cognee.modules.tools.config import get_tools_config
                from cognee.modules.tools.connections import list_tool_connections
                from cognee.modules.tools.text_to_sql import TOOL_NAME, run_text_to_sql

                tools_config = get_tools_config()
                if not tools_config.tool_calls_enabled:
                    # The scope was requested explicitly ("tools" never comes from
                    # "auto"/"all"), so a silent empty result would read as "no
                    # data" — fail loudly instead.
                    raise CogneeValidationError(
                        message=(
                            "Recall scope 'tools' requested but tool calls are disabled. "
                            "Set TOOL_CALLS_ENABLED=true and register a connection with "
                            "cognee.tools.register_sql_connection(...) to enable it."
                        ),
                        name="ToolCallsDisabledError",
                    )

                caller_user_id = await _resolve_user_id(user)
                if not caller_user_id:
                    return []
                user_uuid = _UUID(caller_user_id)

                connection_names = tool_connections or [
                    connection["name"] for connection in await list_tool_connections(user_uuid)
                ]
                if not connection_names:
                    return []

                span.set_attribute("cognee.recall.tool_connections", len(connection_names))

                entries: list[RecallResponse] = []
                for connection_name in connection_names:
                    # Authorization failures (unknown/non-owned name) raise;
                    # generation/execution failures come back as success=False
                    # entries so one dead database never aborts a multi-source
                    # recall.
                    result = await run_text_to_sql(user_uuid, connection_name, query_text)
                    logger.info(
                        "text_to_sql on '%s': success=%s rows=%d attempts=%d",
                        connection_name,
                        result.success,
                        result.row_count,
                        result.attempts,
                    )
                    entries.append(
                        ResponseToolEntry(
                            source="tools",
                            tool_name=TOOL_NAME,
                            question=query_text,
                            text=result.render_text(),
                            success=result.success,
                            error=result.error,
                            structured=result.structured(),
                        )
                    )
                return entries

            async def _run_code() -> list[RecallResponse]:
                nonlocal user

                from cognee.modules.recall.methods.normalize_search_payload import (
                    normalize_search_payload,
                )
                from cognee.modules.retrieval.code_retriever import CodeSeedNotFoundError
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

                # Dataset UUIDs take precedence over names, matching the graph lane.
                search_dataset_ids = dataset_ids or None
                if search_dataset_ids is None and datasets is not None:
                    search_dataset_ids = [
                        dataset.id
                        for dataset in await get_authorized_existing_datasets(
                            datasets, "read", user
                        )
                    ]
                    if not search_dataset_ids:
                        raise DatasetNotFoundError(message="No datasets found.")

                try:
                    code_results = await authorized_search(
                        query_text=query_text,
                        query_type=SearchType.CODE,
                        user=user,
                        dataset_ids=search_dataset_ids,
                        top_k=top_k,
                        retriever_specific_config=code_query,
                    )
                except CodeSeedNotFoundError:
                    # A seed the code graph cannot resolve means "no code facts
                    # for this prompt", not an error — the lane contributes
                    # nothing, like an empty session lane. Invalid operations or
                    # arguments still raise: those are caller bugs.
                    return []

                tagged: list[RecallResponse] = []
                for payload in code_results:
                    completion = getattr(payload, "completion", None)
                    if isinstance(completion, dict) and completion.get("seed_not_found"):
                        # Multi-dataset searches soften per-dataset seed misses
                        # into marker payloads; they carry no facts, so drop
                        # them here for the same reason as the except above.
                        continue
                    items: list[SearchResultItem] = normalize_search_payload(payload)
                    tagged.extend(
                        ResponseCodeEntry(**item.model_dump(), source="code") for item in items
                    )
                return tagged

            async def _run_skill_gate(gate_top_k: int) -> list[RecallResponse]:
                """Metadata-only SKILLS lookup for the deterministic skill gate.

                Skipped silently unless exactly one dataset is targeted (skill
                lookup is single-dataset by invariant). Any failure contributes
                nothing instead of failing the recall.
                """
                from cognee.modules.search.methods.search import authorized_search

                try:
                    gate_user = user
                    if gate_user is None:
                        gate_user = await get_default_user()

                    if dataset_ids and len(dataset_ids) == 1:
                        gate_dataset_ids = list(dataset_ids)
                    elif datasets and len(datasets) == 1:
                        authorized = await get_authorized_existing_datasets(
                            datasets, "read", gate_user
                        )
                        if len(authorized) != 1:
                            return []
                        gate_dataset_ids = [dataset.id for dataset in authorized]
                    else:
                        return []

                    payloads = await authorized_search(
                        query_text=query_text,
                        query_type=SearchType.SKILLS,
                        user=gate_user,
                        dataset_ids=gate_dataset_ids,
                        top_k=gate_top_k,
                    )
                except Exception as error:
                    logger.warning("Skill gate lookup failed (non-fatal): %s", error)
                    return []

                entries: list[RecallResponse] = []
                for payload in payloads or []:
                    for item in getattr(payload, "completion", None) or []:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("name") or ""
                        description = item.get("description") or ""
                        entries.append(
                            ResponseSkillEntry(
                                source="skills",
                                text=f"{name}: {description}" if description else name,
                                skill={k: v for k, v in item.items() if k != "score"},
                                score=item.get("score"),
                            )
                        )
                return entries

            runners = {
                "session": _run_session,
                "trace": _run_trace,
                "session_context": _run_session_context,
                "graph": _run_graph,
                "tools": _run_tools,
                "code": _run_code,
            }

            # Deterministic skill gate: a procedural-looking query triggers a
            # concurrent metadata-only SKILLS lookup (one vector search, no
            # LLM call). Additive only — the main lanes never wait on it, and
            # explicit SKILLS / AGENTIC_COMPLETION calls bypass it.
            skills_task = None
            if (
                "graph" in sources
                and not only_context
                and query_type not in (SearchType.SKILLS, SearchType.AGENTIC_COMPLETION)
            ):
                from cognee.api.v1.recall.skill_gate import (
                    DEFAULT_SKILL_GATE_TOP_K,
                    should_search_skills,
                    skill_gate_enabled,
                )

                if skill_gate_enabled() and should_search_skills(query_text).fired:
                    skills_task = asyncio.create_task(_run_skill_gate(DEFAULT_SKILL_GATE_TOP_K))

            session_result_count = 0
            try:
                for src in sources:
                    runner = runners.get(src)
                    if runner is None:
                        continue
                    # Auto mode special case: session hit short-circuits graph.
                    if auto_fallthrough and src == "graph" and merged:
                        break
                    # on_empty: the other sources gave cognee enough context — don't
                    # go back to the external database.
                    if src == "tools" and tools_trigger == "on_empty" and merged:
                        continue
                    part = await runner()
                    if src == "session":
                        session_result_count = len(part)
                    merged.extend(part)
            except BaseException:
                if skills_task is not None:
                    skills_task.cancel()
                raise

            if skills_task is not None:
                merged.extend(await skills_task)

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
