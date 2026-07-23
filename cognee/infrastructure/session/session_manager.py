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
from cognee.infrastructure.session.session_embeddings import (
    delete_session_qa_vector,
    delete_session_qa_vectors,
    index_session_qa,
)
from cognee.infrastructure.session.session_turn import (
    SessionTurnPreparation,
    generate_session_answer,
    prepare_session_turn as _prepare_turn,
)
from cognee.modules.observability import (
    COGNEE_DATA_SIZE_BYTES,
    COGNEE_SESSION_ENTRY_COUNT,
    COGNEE_SESSION_ID,
    new_span,
)
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.session_lifecycle.metrics import record_session_activity
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger("SessionManager")


class SessionManager:
    """
    Manages session QA entries.
    """

    @staticmethod
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
        self._validate_session_params(user_id=user_id, session_id=session_id)
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
            )
            await index_session_qa(
                user_id=user_id,
                session_id=session_id,
                qa_id=qa_id,
                question=question,
                answer=answer,
            )
            await record_session_activity(user_id, session_id)
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
        self._validate_session_params(user_id=user_id, session_id=session_id)
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
        await record_session_activity(user_id, session_id, errored=status == "error")
        await self._record_trace_step_usage(
            user_id=user_id,
            session_id=session_id,
            memory_query=memory_query,
            memory_context=memory_context,
            method_params=method_params,
            method_return_value=method_return_value,
        )
        await self._maybe_extract_agent_context(
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
            origin_function=origin_function,
            status=status,
            error_message=error_message,
        )
        return trace_id

    async def _record_trace_step_usage(
        self,
        *,
        user_id: str,
        session_id: str,
        memory_query: str,
        memory_context: str,
        method_params: Any,
        method_return_value: Any,
    ) -> None:
        """Estimate this trace step's token usage and accumulate it onto the
        session row. Best-effort — never let usage accounting break a write."""

        def _as_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            try:
                import json

                return json.dumps(value, default=str)
            except (TypeError, ValueError):
                return str(value)

        try:
            from cognee.modules.session_lifecycle.usage_tracking import record_transcript_usage

            await record_transcript_usage(
                session_id=session_id,
                user_id=user_id,
                input_text=f"{memory_query}\n{_as_text(method_params)}",
                output_text=f"{memory_context}\n{_as_text(method_return_value)}",
            )
        except Exception as exc:
            logger.debug("_record_trace_step_usage failed (%s)", exc)

    async def _maybe_extract_agent_context(
        self,
        *,
        user_id: str,
        session_id: str,
        trace_id: str,
        origin_function: str,
        status: str,
        error_message: str,
    ) -> None:
        """Derive agent-profile lessons from a just-stored trace step. Gated and fail-open.

        Runs only when automatic session context is enabled, and never lets an extraction
        failure escape — the trace row is already saved by the time this runs.
        """
        if not self.is_auto_feedback_enabled():
            return
        try:
            from cognee.infrastructure.session.agent_context_extraction import (
                extract_live_agent_context,
                extract_pending_agent_context,
            )

            await extract_live_agent_context(
                session_manager=self,
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
                origin_function=origin_function,
                status=status,
                error_message=error_message,
            )
            await extract_pending_agent_context(
                session_manager=self,
                user_id=user_id,
                session_id=session_id,
            )
        except Exception as error:
            logger.warning("Agent-context extraction skipped: %s", error)

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

    def _session_usage_scope(self, user_id, session_id: str):
        """Return a session-usage tracking context, or a no-op when usage can't be attributed."""
        from contextlib import nullcontext
        from uuid import UUID

        from cognee.modules.session_lifecycle.usage_tracking import track_session_usage

        try:
            usage_uid = UUID(str(user_id)) if user_id is not None else None
        except (ValueError, TypeError):
            usage_uid = None
        if usage_uid is not None and session_id:
            return track_session_usage(session_id, usage_uid)
        return nullcontext()

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
        """Run one session turn under a session-usage scope, then return the answer."""
        user_id = getattr(session_user.get(), "id", None)
        resolved_session_id = self._resolve_session_id(session_id)
        async with self._session_usage_scope(user_id, resolved_session_id):
            return await self._run_session_turn(
                user_id=user_id,
                session_id=resolved_session_id,
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

    async def _run_session_turn(
        self,
        *,
        user_id,
        session_id: str,
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
        """Answer or acknowledge one turn, then record it.

        When session caching is unavailable, runs a plain completion without history and
        does not record. Otherwise: prepare the turn, generate an answer (or take the
        feedback acknowledgement), and store the exchange so every turn stays recallable.
        """
        if not self.is_session_available_for_completion(user_id):
            return await generate_completion(
                query=query,
                context=context,
                user_prompt_path=user_prompt_path,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                response_model=response_model,
            )

        if turn_preparation is None:
            turn_preparation = await self.prepare_session_turn(
                query=query, session_id=session_id, user_id=str(user_id)
            )

        # Every turn — answered or feedback-only — falls through to a single add_qa, so the
        # whole conversation stays in history and vector recall.
        if turn_preparation.should_answer:
            answer_query = (
                (turn_preparation.effective_query or "").strip()
                or (effective_query or "").strip()
                or query
            )
            answer, context_to_store, used_session_context_ids = await generate_session_answer(
                self,
                user_id=str(user_id),
                session_id=session_id,
                answer_query=answer_query,
                context=context,
                user_prompt_path=user_prompt_path,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                response_model=response_model,
                summarize_context=summarize_context,
                max_context_chars=max_context_chars,
            )
            graph_elements = used_graph_element_ids
        else:
            # Feedback-only turn: nothing to answer, but we still record the exchange
            # (question + acknowledgement) so it stays in history and vector recall.
            answer = turn_preparation.response_to_user or "Thanks for your feedback."
            context_to_store = ""
            used_session_context_ids = None
            graph_elements = None

        await self.add_qa(
            user_id=str(user_id),
            question=query,
            context=context_to_store,
            answer=str(answer),
            session_id=session_id,
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
        self._validate_session_params(user_id=user_id, session_id=session_id, last_n=last_n)
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

    async def get_session_entries_by_ids(
        self,
        *,
        user_id: str,
        qa_ids: list[str],
        session_id: str | None = None,
    ) -> list[SessionQAEntry]:
        """Get specific session QA entries by qa_id, returned in chronological order."""
        session_id = self._resolve_session_id(session_id)
        self._validate_session_params(user_id=user_id, session_id=session_id)
        for qa_id in qa_ids:
            self._validate_session_params(qa_id=qa_id)
        if not self.is_available or not qa_ids:
            return []

        return await self._cache.get_qa_entries_by_ids(user_id, session_id, qa_ids)

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
        self._validate_session_params(user_id=user_id, session_id=session_id, last_n=last_n)
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
        self._validate_session_params(user_id=user_id, session_id=session_id, last_n=last_n)
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
        self._validate_session_params(user_id=user_id, session_id=session_id)
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
        self._validate_session_params(user_id=user_id, session_id=session_id, qa_id=qa_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping update_qa")
            return False

        text_changed = question is not None or answer is not None
        async with session_lock(session_id, "update_qa"):
            updated = await self._cache.update_qa_entry(
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
            if not updated:
                return False

            if text_changed:
                entries = await self.get_session_entries_by_ids(
                    user_id=user_id,
                    session_id=session_id,
                    qa_ids=[qa_id],
                )
                await delete_session_qa_vector(qa_id=qa_id)
                if entries:
                    entry = entries[0]
                    await index_session_qa(
                        user_id=user_id,
                        session_id=session_id,
                        qa_id=qa_id,
                        question=entry.question,
                        answer=entry.answer,
                    )
            return True

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
        self._validate_session_params(user_id=user_id, session_id=session_id, qa_id=qa_id)
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
        self._validate_session_params(user_id=user_id, session_id=session_id, qa_id=qa_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_qa")
            return False

        async with session_lock(session_id, "update_qa"):
            deleted = await self._cache.delete_qa_entry(
                user_id=user_id,
                session_id=session_id,
                qa_id=qa_id,
            )
            if deleted:
                await delete_session_qa_vector(qa_id=qa_id)
            return deleted

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

        Raises SessionParameterValidationError for invalid user_id/session_id.
        Fail-open on infrastructure errors: returns False when the cache is
        unavailable or the cache operation fails.
        """
        session_id = self._resolve_session_id(session_id)
        self._validate_session_params(user_id=user_id, session_id=session_id)
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

        Raises SessionParameterValidationError for invalid user_id/session_id.
        Fail-open on infrastructure errors: returns [] when the cache is
        unavailable or the cache operation fails.
        """
        session_id = self._resolve_session_id(session_id)
        self._validate_session_params(user_id=user_id, session_id=session_id)
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

        Raises SessionParameterValidationError for invalid user_id/session_id.
        Fail-open on infrastructure errors: returns False when the cache is
        unavailable or the cache operation fails.
        """
        session_id = self._resolve_session_id(session_id)
        self._validate_session_params(user_id=user_id, session_id=session_id)
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

        Raises SessionParameterValidationError for invalid user_id/session_id.
        Fail-open on infrastructure errors: returns False when the cache is
        unavailable or the cache operation fails.
        """
        session_id = self._resolve_session_id(session_id)
        self._validate_session_params(user_id=user_id, session_id=session_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_session_context")
            return False
        try:
            return await self._cache.delete_session_context(user_id, session_id)
        except Exception as e:
            logger.warning("SessionManager: delete_session_context failed: %s", e)
            return False

    async def delete_session(self, *, user_id: str, session_id: str | None = None) -> bool:
        """
        Delete the entire session and all its QA entries.

        Returns True if deleted, False if session did not exist or cache unavailable.
        """
        session_id = self._resolve_session_id(session_id)
        self._validate_session_params(user_id=user_id, session_id=session_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_session")
            return False

        # One-release cleanup for graph snapshots written by the removed
        # graph-to-session sync feature.
        graph_key = f"graph_knowledge:{user_id}:{session_id}"
        try:
            await self._cache.delete_value(graph_key)
        except (NotImplementedError, AttributeError, TypeError):
            # Adapter predates the KV interface (missing, non-async, or
            # different-signature delete_value), fall back to legacy duck-typing
            try:
                await self._cache.async_redis.delete(graph_key)
            except AttributeError:
                try:
                    del self._cache._cache[graph_key]
                except Exception:
                    pass
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

        deleted = await self._cache.delete_session(
            user_id=user_id,
            session_id=session_id,
        )
        if deleted:
            await delete_session_qa_vectors(user_id=user_id, session_id=session_id)
        return deleted
