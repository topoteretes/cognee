import uuid
from typing import Any, Optional, Type, Union

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.databases.exceptions import SessionParameterValidationError
from cognee.modules.retrieval.utils.completion import (
    generate_completion,
    generate_session_completion_with_optional_summary,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("SessionManager")


def _validate_session_params(
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    qa_id: Optional[str] = None,
    last_n: Optional[int] = None,
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
        cache_engine,
        default_session_id: str = "default_session",
        session_history_last_n: int = 10,
    ):
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

    def _resolve_session_id(self, session_id: Optional[str]) -> str:
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
        session_id: Optional[str] = None,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
    ) -> Optional[str]:
        """
        Add a QA to the session. Returns qa_id, or None if cache unavailable.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping add_qa")
            return None

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
        )
        return qa_id

    def is_session_available_for_completion(self, user_id: Optional[str]) -> bool:
        """Return True if session (history + save) is available for completion."""
        if not user_id or not self.is_available:
            return False
        cache_config = CacheConfig()
        return bool(cache_config.caching)

    async def _get_formatted_history(self, user_id: str, session_id: str) -> str:
        """Load session and return formatted conversation history string."""
        history: Union[str, list] = await self.get_session(
            user_id=user_id,
            session_id=session_id,
            formatted=True,
            last_n=self.session_history_last_n,
            include_context=False,
        )
        return history if isinstance(history, str) else ""

    async def generate_completion_with_session(
        self,
        *,
        session_id: Optional[str] = None,
        query: str,
        context: str,
        user_prompt_path: str,
        system_prompt_path: str,
        system_prompt: Optional[str] = None,
        response_model: Type = str,
        summarize_context: bool = False,
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
        conversation_history = await self._get_formatted_history(str(user_id), resolved_session_id)

        cache_config = CacheConfig()
        run_auto_feedback = cache_config.caching and cache_config.auto_feedback

        last_qa_id: Optional[str] = None
        if run_auto_feedback:
            entries = await self.get_session(
                user_id=str(user_id),
                session_id=resolved_session_id,
                formatted=False,
                last_n=1,
            )
            if isinstance(entries, list) and entries:
                last_entry = entries[-1]
                last_qa_id = (
                    last_entry.get("qa_id")
                    if isinstance(last_entry, dict)
                    else None
                )

        completion, context_to_store, feedback_result = await generate_session_completion_with_optional_summary(
            query=query,
            context=context,
            conversation_history=conversation_history,
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            response_model=response_model,
            summarize_context=summarize_context,
            run_feedback_detection=run_auto_feedback,
        )

        feedback_detected = (
            run_auto_feedback
            and feedback_result is not None
            and feedback_result.feedback_detected
            and last_qa_id is not None
        )

        if feedback_detected:
            try:
                score: Optional[int] = None
                if feedback_result.feedback_score is not None:
                    s = float(feedback_result.feedback_score)
                    score = int(round(min(5, max(1, s))))
                feedback_text = (feedback_result.feedback_text or "").strip()
                if not feedback_text:
                    feedback_text = f"User message: {query.strip()[:500]}"
                await self.add_feedback(
                    user_id=str(user_id),
                    session_id=resolved_session_id,
                    qa_id=last_qa_id,
                    feedback_text=feedback_text,
                    feedback_score=score,
                )
            except Exception as e:
                logger.warning(
                    "Auto-feedback persistence failed, proceeding without storing feedback: %s",
                    e,
                    exc_info=False,
                )
            response = (feedback_result.response_to_user or "").strip()
            return response if response else "Thanks for your feedback."

        await self.add_qa(
            user_id=str(user_id),
            question=query,
            context=context_to_store,
            answer=str(completion),
            session_id=resolved_session_id,
        )
        return completion

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
        last_n: Optional[int] = None,
        formatted: bool = False,
        session_id: Optional[str] = None,
        include_context: bool = True,
    ) -> Union[list[dict], str]:
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

        if last_n is not None:
            entries = await self._cache.get_latest_qa_entries(user_id, session_id, last_n=last_n)
        else:
            entries = await self._cache.get_all_qa_entries(user_id, session_id)

        if entries is None:
            return "" if formatted else []
        entries_list = list(entries)
        return (
            self.format_entries(entries_list, include_context=include_context)
            if formatted
            else entries_list
        )

    async def update_qa(
        self,
        *,
        user_id: str,
        qa_id: str,
        question: Optional[str] = None,
        context: Optional[str] = None,
        answer: Optional[str] = None,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Update a QA entry by qa_id.

        Only passed fields are updated; None preserves existing values.
        Returns True if updated, False if not found or cache unavailable.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id, qa_id=qa_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping update_qa")
            return False

        return await self._cache.update_qa_entry(
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
            question=question,
            context=context,
            answer=answer,
            feedback_text=feedback_text,
            feedback_score=feedback_score,
        )

    async def add_feedback(
        self,
        *,
        user_id: str,
        qa_id: str,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Add or update feedback for a QA entry.

        Convenience method that updates only feedback fields.
        Returns True if updated, False if not found or cache unavailable.
        """
        return await self.update_qa(
            user_id=user_id,
            qa_id=qa_id,
            feedback_text=feedback_text,
            feedback_score=feedback_score,
            session_id=session_id,
        )

    async def delete_feedback(
        self,
        *,
        user_id: str,
        qa_id: str,
        session_id: Optional[str] = None,
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
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a single QA entry by qa_id.

        Returns True if deleted, False if not found or cache unavailable.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id, qa_id=qa_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_qa")
            return False

        return await self._cache.delete_qa_entry(
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
        )

    async def delete_session(self, *, user_id: str, session_id: Optional[str] = None) -> bool:
        """
        Delete the entire session and all its QA entries.

        Returns True if deleted, False if session did not exist or cache unavailable.
        """
        session_id = self._resolve_session_id(session_id)
        _validate_session_params(user_id=user_id, session_id=session_id)
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_session")
            return False

        return await self._cache.delete_session(
            user_id=user_id,
            session_id=session_id,
        )
