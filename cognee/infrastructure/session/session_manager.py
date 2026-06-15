import uuid
from typing import Any

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache import SessionAgentTraceEntry, SessionQAEntry
from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter
from cognee.infrastructure.databases.exceptions import SessionParameterValidationError
from cognee.infrastructure.session.session_agent_trace import (
    fallback_agent_trace_feedback,
    generate_agent_trace_feedback,
)
from cognee.infrastructure.session.session_embeddings import embed_text_safe
from cognee.infrastructure.session.session_turn import (
    SessionTurnPreparation,
    build_active_context_block_safe,
    compose_session_prompt,
    prepare_session_turn as _prepare_turn,
    select_session_history,
)
from cognee.modules.observability import (
    COGNEE_DATA_SIZE_BYTES,
    COGNEE_SESSION_ENTRY_COUNT,
    COGNEE_SESSION_ID,
    new_span,
)
from cognee.modules.retrieval.utils.completion import (
    generate_completion,
    generate_session_completion_with_optional_summary,
)
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger("SessionManager")


_session_record_write_failed = False


async def _record_session_activity(
    user_id: str,
    session_id: str,
    *,
    errored: bool = False,
) -> None:
    """Write a lifecycle heartbeat for this session.

    Upserts + touches the SessionRecord row in one DB round trip.
    Swallows failures — the session_records table is optional for
    SessionManager correctness — but logs once at WARNING per process
    so silent breakage is visible in ops.
    """
    global _session_record_write_failed

    try:
        from uuid import UUID

        from cognee.modules.session_lifecycle.metrics import (
            accumulate_usage,
            ensure_and_touch_session,
        )

        try:
            user_uuid = UUID(str(user_id))
        except (ValueError, TypeError):
            return

        await ensure_and_touch_session(session_id=session_id, user_id=user_uuid)
        if errored:
            await accumulate_usage(session_id=session_id, user_id=user_uuid, errored=True)
    except Exception as exc:
        if not _session_record_write_failed:
            _session_record_write_failed = True
            logger.warning(
                "SessionManager: session_records write failed (%s); "
                "subsequent failures will log at debug. "
                "Check alembic migrations for the session_records table.",
                exc,
            )
        else:
            logger.debug("SessionManager: session_records write failed (%s)", exc)


def _validate_session_params(
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    qa_id: str | None = None,
    last_n: int | None = None,
) -> None:
    """
    Validate session parameters. Raises SessionParameterValidationError if any
    provided parameter is invalid.

    - user_id, session_id, qa_id: must be non-empty strings when provided.
    - last_n: when provided, must be a positive integer.
    """
    checks = (
        (user_id, "user_id"),
        (session_id, "session_id"),
        (qa_id, "qa_id"),
    )
    for value, name in checks:
        if value is not None and (not str(value).strip()):
            raise SessionParameterValidationError(message=f"{name} must be a non-empty string")
    if last_n is not None and (not isinstance(last_n, int) or last_n < 1):
        raise SessionParameterValidationError(message="last_n must be a positive integer")


class SessionManager:
    """
    Manages session QA entries.
    """

    def __init__(
        self,
        # TODO: this type should be 'CacheDBInterface', but the current code doesn't use this
        # interface and instead calls functions specific to its implementations.
        cache_engine: Any,
        default_session_id: str = "default_session",
        session_history_last_n: int = 10,
    ) -> None:
        """
        Initialize SessionManager with a cache engine.

        Args:
            cache_engine: CacheDBInterface implementation (RedisAdapter or FsCacheAdapter).
                         Can be None if caching is disabled.
            default_session_id: Session ID to use when session_id is None. Defaults to
                               "default_session".
            session_history_last_n: Number of prior Q&A entries to include in conversation
                                   history for completion. Defaults to 10.
        """
        self._cache = cache_engine
        self.default_session_id = default_session_id
        self.session_history_last_n = session_history_last_n

    def _resolve_session_id(self, session_id: str | None) -> str:
        """Return session_id if provided, otherwise default_session_id."""
        return session_id if session_id is not None else self.default_session_id

    @property
    def is_available(self) -> bool:
        """Return True if the cache engine is available."""
        return self._cache is not None

    async def add_qa(
        self,
        *,
        user_id: str,
        question: str,
        context: str,
        answer: str,
        session_id: str | None = None,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
        used_graph_element_ids: dict | None = None,
        used_session_context_ids: list | None = None,
    ) -> str | None:
        """
        Add a QA to the session. Returns qa_id, or None if cache unavailable.
        used_graph_element_ids: Optional dict with keys "node_ids" and "edge_ids" (lists of str).
        used_session_context_ids: Optional list of session-context entry ids served to this answer.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping add_qa")
            return None

        data_size = len(answer.encode("utf-8", errors="replace")) if answer else 0
        data_size += len(question.encode("utf-8", errors="replace")) if question else 0
        data_size += len(context.encode("utf-8", errors="replace")) if context else 0

        with new_span("cognee.session.add_qa") as span:
            span.set_attribute(COGNEE_SESSION_ID, session_id)
            span.set_attribute(COGNEE_DATA_SIZE_BYTES, data_size)

            send_telemetry(
                "cognee.session.add_qa",
                user_id,
                additional_properties={
                    "session_id": session_id,
                    "data_size_bytes": data_size,
                    "has_feedback": feedback_score is not None,
                    "has_graph_elements": used_graph_element_ids is not None,
                },
            )

            qa_id = str(uuid.uuid4())
            embedding = await embed_text_safe(f"{question or ''}\n{answer or ''}")
            await self._cache.create_qa_entry(
                user_id=user_id,
                session_id=session_id,
                qa_id=qa_id,
                question=question,
                context=context,
                answer=answer,
                feedback_text=feedback_text,
                feedback_score=feedback_score,
                used_graph_element_ids=used_graph_element_ids,
                used_session_context_ids=used_session_context_ids,
                embedding=embedding,
            )
            await _record_session_activity(user_id, session_id)
            return qa_id

    async def add_agent_trace_step(
        self,
        *,
        user_id: str,
        origin_function: str,
        status: str,
        generate_feedback_with_llm: bool = True,
        session_id: str | None = None,
        memory_query: str = "",
        memory_context: str = "",
        method_params: dict | None = None,
        method_return_value: Any = None,
        error_message: str = "",
    ) -> str | None:
        """
        Append one agent trace step to the session trace payload.

        Returns trace_id, or None if cache unavailable.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping add_agent_trace_step")
            return None

        trace_id = str(uuid.uuid4())
        if generate_feedback_with_llm:
            session_feedback = await generate_agent_trace_feedback(
                origin_function=origin_function,
                status=status,
                method_return_value=method_return_value,
                error_message=error_message,
            )
        else:
            session_feedback = fallback_agent_trace_feedback(
                origin_function=origin_function,
                status=status,
                error_message=error_message,
            )
        await self._cache.append_agent_trace_step(
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
            origin_function=origin_function,
            status=status,
            memory_query=memory_query,
            memory_context=memory_context,
            method_params=method_params,
            method_return_value=method_return_value,
            error_message=error_message,
            session_feedback=session_feedback,
        )
        await _record_session_activity(user_id, session_id, errored=status == "error")
        return trace_id

    def is_session_available_for_completion(self, user_id: str | None) -> bool:
        """Return True if session (history + save) is available for completion."""
        if not user_id or not self.is_available:
            return False
        cache_config = CacheConfig()
        return bool(cache_config.caching)

    def is_auto_feedback_enabled(self) -> bool:
        """Return True if caching and automatic turn-feedback analysis are both enabled."""
        cache_config = CacheConfig()
        return bool(cache_config.caching and cache_config.auto_feedback)

    async def prepare_session_turn(
        self,
        *,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> SessionTurnPreparation:
        """Analyze one user turn before retrieval/answer generation.

        Thin delegate to ``session_turn.prepare_session_turn``; see that module for the logic.
        """
        return await _prepare_turn(self, query=query, session_id=session_id, user_id=user_id)

    async def generate_completion_with_session(
        self,
        *,
        session_id: str | None = None,
        query: str,
        context: str,
        user_prompt_path: str,
        system_prompt_path: str,
        system_prompt: str | None = None,
        response_model: type = str,
        summarize_context: bool = False,
        used_graph_element_ids: dict | None = None,
        max_context_chars: int | None = None,
        effective_query: str | None = None,
        turn_preparation: SessionTurnPreparation | None = None,
    ) -> Any:
        from uuid import UUID as _UUID

        from cognee.modules.session_lifecycle.usage_tracking import track_session_usage

        _ctx_user = session_user.get()
        _ctx_uid_raw = getattr(_ctx_user, "id", None)
        _ctx_sid = self._resolve_session_id(session_id)
        try:
            _ctx_uid = _UUID(str(_ctx_uid_raw)) if _ctx_uid_raw is not None else None
        except (ValueError, TypeError):
            _ctx_uid = None

        if _ctx_uid is not None and _ctx_sid:
            async with track_session_usage(_ctx_sid, _ctx_uid):
                return await self._generate_completion_with_session_inner(
                    session_id=session_id,
                    query=query,
                    context=context,
                    user_prompt_path=user_prompt_path,
                    system_prompt_path=system_prompt_path,
                    system_prompt=system_prompt,
                    response_model=response_model,
                    summarize_context=summarize_context,
                    used_graph_element_ids=used_graph_element_ids,
                    max_context_chars=max_context_chars,
                    effective_query=effective_query,
                    turn_preparation=turn_preparation,
                )
        return await self._generate_completion_with_session_inner(
            session_id=session_id,
            query=query,
            context=context,
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            response_model=response_model,
            summarize_context=summarize_context,
            used_graph_element_ids=used_graph_element_ids,
            max_context_chars=max_context_chars,
            effective_query=effective_query,
            turn_preparation=turn_preparation,
        )

    async def _generate_completion_with_session_inner(
        self,
        *,
        session_id: str | None = None,
        query: str,
        context: str,
        user_prompt_path: str,
        system_prompt_path: str,
        system_prompt: str | None = None,
        response_model: type = str,
        summarize_context: bool = False,
        used_graph_element_ids: dict | None = None,
        max_context_chars: int | None = None,
        effective_query: str | None = None,
        turn_preparation: SessionTurnPreparation | None = None,
    ) -> Any:
        """
        Run single-query completion with session: read history, generate, save QA.

        Resolves user_id from session_user; if no user or caching disabled, runs
        completion without history and does not save. Otherwise gets formatted
        history, runs one or two LLM calls depending on summarize_context,
        saves via add_qa, and returns the completion.

        Args:
            session_id: Session identifier; defaults to default_session_id if None.
            query: User question.
            context: Retrieved context for the completion.
            user_prompt_path: Path for user prompt template.
            system_prompt_path: Path for system prompt template.
            system_prompt: Optional override system prompt.
            response_model: Pydantic model or type for structured output (default str).
            summarize_context: If True, run summarization LLM call and store summary
                in QA context; if False, single LLM call and store "" for context.

        Returns:
            (completion, qa_id): completion from LLM; qa_id if saved, None otherwise.
        """
        user = session_user.get()
        user_id = getattr(user, "id", None)

        if not self.is_session_available_for_completion(user_id):
            return await generate_completion(
                query=query,
                context=context,
                user_prompt_path=user_prompt_path,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                response_model=response_model,
            )

        resolved_session_id = self._resolve_session_id(session_id)
        if turn_preparation is None:
            turn_preparation = await self.prepare_session_turn(
                query=query,
                session_id=resolved_session_id,
                user_id=str(user_id),
            )
        # Every turn — answered or feedback-only — falls through to a single add_qa, so the
        # whole conversation stays in history and semantic recall.
        if turn_preparation.should_answer:
            answer_query = (
                (turn_preparation.effective_query or "").strip()
                or (effective_query or "").strip()
                or query
            )
            # One embedding per turn, shared by hybrid history retrieval and context ranking.
            query_embedding = await embed_text_safe(answer_query)
            conversation_history = await select_session_history(
                self, str(user_id), resolved_session_id, query_embedding=query_embedding
            )

            cache_config = CacheConfig()
            served_ids: list[str] = []
            active_context_block = ""
            if cache_config.caching and cache_config.auto_feedback:
                active_context_block, served_ids = await build_active_context_block_safe(
                    self,
                    user_id=str(user_id),
                    session_id=resolved_session_id,
                    query=answer_query,
                    query_embedding=query_embedding,
                )

            # Graph-knowledge snapshot (from improve() sync). Its char budget is separate
            # from the context block's: explicit param > config > unlimited.
            graph_context = await self.get_graph_context(
                user_id=str(user_id), session_id=resolved_session_id
            )
            if graph_context:
                char_limit = max_context_chars
                if char_limit is None:
                    char_limit = cache_config.max_session_context_chars
                if char_limit is not None:
                    graph_context = graph_context[:char_limit]

            conversation_history = compose_session_prompt(
                active_context_block, graph_context, conversation_history
            )

            (
                answer,
                context_to_store,
                _feedback_result,
            ) = await generate_session_completion_with_optional_summary(
                query=answer_query,
                context=context,
                conversation_history=conversation_history,
                user_prompt_path=user_prompt_path,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                response_model=response_model,
                summarize_context=summarize_context,
            )
            used_session_context_ids = served_ids or None
            graph_elements = used_graph_element_ids
        else:
            # Feedback-only turn: nothing to answer, but we still record the exchange
            # (question + acknowledgement) so it stays in history and semantic recall.
            answer = turn_preparation.response_to_user or "Thanks for your feedback."
            context_to_store = ""
            used_session_context_ids = None
            graph_elements = None

        await self.add_qa(
            user_id=str(user_id),
            question=query,
            context=context_to_store,
            answer=str(answer),
            session_id=resolved_session_id,
            used_graph_element_ids=graph_elements,
            used_session_context_ids=used_session_context_ids,
        )
        return answer

    @staticmethod
    def format_entries(entries: list[dict], include_context: bool = True) -> str:
        """
        Format QA entries as a string for LLM prompt context.

        Args:
            entries: List of QA entry dicts (question, context, answer, time, etc.).
            include_context: If True, include CONTEXT line for each entry; if False, omit it.
                            Default True. Use False when building conversation history for completion.
        """
        if not entries:
            return ""
        lines = ["Previous conversation:\n\n"]
        for entry in entries:
            lines.append(f"[{entry.get('time', 'Unknown time')}]\n")
            lines.append(f"QUESTION: {entry.get('question', '')}\n")
            if include_context:
                lines.append(f"CONTEXT: {entry.get('context', '')}\n")
            lines.append(f"ANSWER: {entry.get('answer', '')}\n\n")
        return "".join(lines)

    async def get_session(
        self,
        *,
        user_id: str,
        last_n: int | None = None,
        formatted: bool = False,
        session_id: str | None = None,
        include_context: bool = True,
    ) -> list[SessionQAEntry] | str:
        """
        Get session QAs by (user_id, session_id).

        Args:
            user_id: User identifier.
            last_n: If set, return only the last N entries. Otherwise return all.
            formatted: If True, return prompt-formatted string; if False, return list of entry dicts.
            session_id: Session identifier. Defaults to default_session_id if None.
            include_context: When formatted=True, include CONTEXT in each entry. Default True.
                            Set False for conversation history used in completion prompts.

        Returns:
            List of QA entry dicts, or formatted string if formatted=True.
            Empty list or empty string if cache unavailable or session not found.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id, last_n=last_n)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, returning empty session")
            return "" if formatted else []

        with new_span("cognee.session.get_session") as span:
            span.set_attribute(COGNEE_SESSION_ID, session_id)

            if last_n is not None:
                entries = await self._cache.get_latest_qa_entries(
                    user_id, session_id, last_n=last_n
                )
            else:
                entries = await self._cache.get_all_qa_entries(user_id, session_id)

            entry_count = len(entries) if entries else 0
            span.set_attribute(COGNEE_SESSION_ENTRY_COUNT, entry_count)

        if entries is None:
            return "" if formatted else []
        entries_list = list(entries)
        return (
            self.format_entries(
                [entry.model_dump() for entry in entries_list], include_context=include_context
            )
            if formatted
            else entries_list
        )

    async def get_agent_trace_session(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        last_n: int | None = None,
    ) -> list[SessionAgentTraceEntry]:
        """
        Get the agent trace session for the given user/session pair.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id, last_n=last_n)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, returning empty agent trace session")
            return []

        entries = await self._cache.get_agent_trace_session(user_id, session_id, last_n=last_n)
        return entries

    async def get_agent_trace_feedback(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        last_n: int | None = None,
    ) -> list[str]:
        """
        Get only per-step feedback strings for the trace session.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id, last_n=last_n)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, returning empty agent trace feedback")
            return []

        feedback_list = await self._cache.get_agent_trace_feedback(
            user_id, session_id, last_n=last_n
        )
        return list(feedback_list) if feedback_list else []

    async def get_agent_trace_count(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> int:
        """
        Get the number of trace steps stored for the given user/session pair.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, returning empty agent trace count")
            return 0

        return await self._cache.get_agent_trace_count(user_id, session_id)

    async def update_qa(
        self,
        *,
        user_id: str,
        qa_id: str,
        question: str | None = None,
        context: str | None = None,
        answer: str | None = None,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
        used_graph_element_ids: dict | None = None,
        memify_metadata: dict | None = None,
        used_session_context_ids: list | None = None,
        session_id: str | None = None,
    ) -> bool:
        """
        Update a QA entry by qa_id.

        Only passed fields are updated; None preserves existing values.
        Returns True if updated, False if not found or cache unavailable.
        memify_metadata: Optional dict with status keys (e.g. "feedback_weights_applied") and bool values.
        used_graph_element_ids: Optional dict with "node_ids" and "edge_ids" lists for frequency weights.
        used_session_context_ids: Optional list of session-context entry ids served to this answer.
        """
        from cognee.infrastructure.locks import session_lock

        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id, qa_id=qa_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping update_qa")
            return False

        async with session_lock(session_id, "update_qa"):
            return await self._cache.update_qa_entry(
                user_id=user_id,
                session_id=session_id,
                qa_id=qa_id,
                question=question,
                context=context,
                answer=answer,
                feedback_text=feedback_text,
                feedback_score=feedback_score,
                used_graph_element_ids=used_graph_element_ids,
                memify_metadata=memify_metadata,
                used_session_context_ids=used_session_context_ids,
            )

    async def add_feedback(
        self,
        *,
        user_id: str,
        qa_id: str,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
        session_id: str | None = None,
    ) -> bool:
        """
        Add or update feedback for a QA entry.

        Convenience method that updates only feedback fields.
        Resets feedback-weight memify status so updated feedback can be re-applied.
        Returns True if updated, False if not found or cache unavailable.
        """
        from cognee.tasks.memify.feedback_weights_constants import (
            MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
        )

        return await self.update_qa(
            user_id=user_id,
            qa_id=qa_id,
            feedback_text=feedback_text,
            feedback_score=feedback_score,
            memify_metadata={MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY: False},
            session_id=session_id,
        )

    async def delete_feedback(
        self,
        *,
        user_id: str,
        qa_id: str,
        session_id: str | None = None,
    ) -> bool:
        """
        Clear feedback for a QA entry (sets feedback_text and feedback_score to None).

        Returns True if updated, False if not found or cache unavailable.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id, qa_id=qa_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_feedback")
            return False

        return await self._cache.delete_feedback(
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
        )

    async def delete_qa(
        self,
        *,
        user_id: str,
        qa_id: str,
        session_id: str | None = None,
    ) -> bool:
        """
        Delete a single QA entry by qa_id.

        Returns True if deleted, False if not found or cache unavailable.
        """
        from cognee.infrastructure.locks import session_lock

        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id, qa_id=qa_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_qa")
            return False

        async with session_lock(session_id, "update_qa"):
            return await self._cache.delete_qa_entry(
                user_id=user_id,
                session_id=session_id,
                qa_id=qa_id,
            )

    # -- Session context entries (active guidance layer) --------------------

    async def create_session_context_entry(
        self,
        *,
        user_id: str,
        entry_dump: dict,
        session_id: str | None = None,
    ) -> bool:
        """
        Append one session-context entry (a plain dict carrying a "kind" field).

        Fail-open: returns False when cache unavailable or on any error, never raises.
        """
        session_id = self._resolve_session_id(session_id)
        try:
            _validate_session_params(user_id=user_id, session_id=session_id)
        except Exception:
            return False
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping create_session_context_entry")
            return False
        try:
            await self._cache.create_session_context_entry(user_id, session_id, entry_dump)
            return True
        except Exception as e:
            logger.warning("SessionManager: create_session_context_entry failed: %s", e)
            return False

    async def get_session_context_entries(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> list[dict]:
        """
        Return all stored session-context entries (both "context" and "feedback" kinds).

        Fail-open: returns [] when cache unavailable or on any error, never raises.
        """
        session_id = self._resolve_session_id(session_id)
        try:
            _validate_session_params(user_id=user_id, session_id=session_id)
        except Exception:
            return []
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, returning empty session context")
            return []
        try:
            return await self._cache.get_session_context_entries(user_id, session_id)
        except Exception as e:
            logger.warning("SessionManager: get_session_context_entries failed: %s", e)
            return []

    async def update_session_context_entry(
        self,
        *,
        user_id: str,
        entry_id: str,
        merge: dict,
        session_id: str | None = None,
    ) -> bool:
        """
        Shallow-merge updates into the session-context entry matching entry["id"].

        Fail-open: returns False when cache unavailable or on any error, never raises.
        """
        session_id = self._resolve_session_id(session_id)
        try:
            _validate_session_params(user_id=user_id, session_id=session_id)
        except Exception:
            return False
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping update_session_context_entry")
            return False
        try:
            return await self._cache.update_session_context_entry(
                user_id, session_id, entry_id, merge
            )
        except Exception as e:
            logger.warning("SessionManager: update_session_context_entry failed: %s", e)
            return False

    async def delete_session_context(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> bool:
        """
        Delete the entire session-context list for the given session.

        Fail-open: returns False when cache unavailable or on any error, never raises.
        """
        session_id = self._resolve_session_id(session_id)
        try:
            _validate_session_params(user_id=user_id, session_id=session_id)
        except Exception:
            return False
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_session_context")
            return False
        try:
            return await self._cache.delete_session_context(user_id, session_id)
        except Exception as e:
            logger.warning("SessionManager: delete_session_context failed: %s", e)
            return False

    # -- Graph knowledge context (separate from QA history) -----------------

    @staticmethod
    def _graph_context_key(user_id: str, session_id: str) -> str:
        """Build the cache key used for session-scoped graph knowledge snapshots."""
        return f"graph_knowledge:{user_id}:{session_id}"

    async def get_graph_context(self, *, user_id: str, session_id: str | None = None) -> str:
        """Return the graph knowledge snapshot for this session, or empty string."""
        if not self.is_available:
            return ""
        session_id = self._resolve_session_id(session_id)
        key = self._graph_context_key(user_id, session_id)
        try:
            raw = await self._cache.async_redis.get(key)
            if raw:
                return raw.decode() if isinstance(raw, bytes) else raw
        except AttributeError:
            # FsCacheAdapter
            try:
                raw = self._cache._cache.get(key)
                if raw:
                    return raw
            except Exception:
                pass
        except Exception:
            pass
        return ""

    async def set_graph_context(
        self, *, user_id: str, session_id: str | None = None, context: str
    ) -> None:
        """Store (or overwrite) the graph knowledge snapshot for this session."""
        if not self.is_available:
            return
        session_id = self._resolve_session_id(session_id)
        key = self._graph_context_key(user_id, session_id)
        try:
            await self._cache.async_redis.set(key, context)
            if self._cache.session_ttl_seconds:
                await self._cache.async_redis.expire(key, self._cache.session_ttl_seconds)
        except AttributeError:
            try:
                self._cache._cache.set(key, context)
            except Exception:
                pass

    async def delete_session(self, *, user_id: str, session_id: str | None = None) -> bool:
        """
        Delete the entire session and all its QA entries.

        Returns True if deleted, False if session did not exist or cache unavailable.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_session")
            return False

        # Also clean up the graph knowledge context key
        graph_key = self._graph_context_key(user_id, session_id)
        try:
            await self._cache.async_redis.delete(graph_key)
        except AttributeError:
            try:
                del self._cache._cache[graph_key]
            except Exception:
                pass
        except Exception:
            pass

        # Also clear the active session-context list (fail-open; adapter.delete_session may also
        # clear it, but this guarantees no leak if the adapter does not).
        try:
            await self._cache.delete_session_context(user_id, session_id)
        except Exception:
            pass

        return await self._cache.delete_session(
            user_id=user_id,
            session_id=session_id,
        )
