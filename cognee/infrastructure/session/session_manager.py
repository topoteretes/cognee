import uuid
from typing import Optional, Union

from cognee.infrastructure.databases.exceptions import SessionParameterValidationError
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

    def __init__(self, cache_engine, default_session_id: str = "default_session"):
        """
        Initialize SessionManager with a cache engine.

        Args:
            cache_engine: CacheDBInterface implementation (RedisAdapter or FsCacheAdapter).
                         Can be None if caching is disabled.
            default_session_id: Session ID to use when session_id is None. Defaults to
                               "default_session".
        """
        self._cache = cache_engine
        self.default_session_id = default_session_id

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
        ttl: Optional[int] = 86400,
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
            ttl=ttl,
        )
        return qa_id

    @staticmethod
    def format_entries(entries: list[dict]) -> str:
        """
        Format QA entries as a string for LLM prompt context.
        """
        if not entries:
            return ""
        lines = ["Previous conversation:\n\n"]
        for entry in entries:
            lines.append(f"[{entry.get('time', 'Unknown time')}]\n")
            lines.append(f"QUESTION: {entry.get('question', '')}\n")
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
    ) -> Union[list[dict], str]:
        """
        Get session QAs by (user_id, session_id).

        Args:
            user_id: User identifier.
            last_n: If set, return only the last N entries. Otherwise return all.
            formatted: If True, return prompt-formatted string; if False, return list of entry dicts.
            session_id: Session identifier. Defaults to default_session_id if None.

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
        return self.format_entries(entries_list) if formatted else entries_list

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

    async def delete_session(
        self, *, user_id: str, session_id: Optional[str] = None
    ) -> bool:
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
