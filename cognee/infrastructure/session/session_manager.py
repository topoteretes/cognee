import uuid
from typing import Optional, Union

from cognee.shared.logging_utils import get_logger

logger = get_logger("SessionManager")


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

    def normalize_session_id(self, session_id: Optional[str]) -> str:
        """Return session_id if provided, otherwise default_session_id."""
        return session_id if session_id is not None else self.default_session_id

    @property
    def is_available(self) -> bool:
        """Return True if the cache engine is available."""
        return self._cache is not None

    async def add_qa(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
        ttl: Optional[int] = 86400,
    ) -> Optional[str]:
        """
        Add a QA to the session. Returns qa_id, or None if cache unavailable.
        """
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
        user_id: str,
        session_id: str,
        last_n: Optional[int] = None,
        formatted: bool = False,
    ) -> Union[list[dict], str]:
        """
        Get session QAs by (user_id, session_id).

        Args:
            user_id: User identifier.
            session_id: Session identifier.
            last_n: If set, return only the last N entries. Otherwise return all.
            formatted: If True, return prompt-formatted string; if False, return list of entry dicts.

        Returns:
            List of QA entry dicts, or formatted string if formatted=True.
            Empty list or empty string if cache unavailable or session not found.
        """
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

    async def get_single_entry(
        self,
        user_id: str,
        session_id: str,
        qa_id: str,
    ) -> Optional[dict]:
        """
        Get a single QA entry by qa_id.

        Args:
            user_id: User identifier.
            session_id: Session identifier.
            qa_id: QA entry identifier.

        Returns:
            The QA entry dict if found, None if not found or cache unavailable.
        """
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, returning None for get_single_entry")
            return None

        entries = await self._cache.get_all_qa_entries(user_id, session_id)
        if entries is None:
            return None
        for entry in entries:
            if entry.get("qa_id") == qa_id:
                return entry
        return None

    async def update_qa(
        self,
        user_id: str,
        session_id: str,
        qa_id: str,
        question: Optional[str] = None,
        context: Optional[str] = None,
        answer: Optional[str] = None,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
    ) -> bool:
        """
        Update a QA entry by qa_id.

        Only passed fields are updated; None preserves existing values.
        Returns True if updated, False if not found or cache unavailable.
        """
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
        user_id: str,
        session_id: str,
        qa_id: str,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
    ) -> bool:
        """
        Add or update feedback for a QA entry.

        Convenience method that updates only feedback fields.
        Returns True if updated, False if not found or cache unavailable.
        """
        return await self.update_qa(
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
            feedback_text=feedback_text,
            feedback_score=feedback_score,
        )

    async def delete_feedback(
        self,
        user_id: str,
        session_id: str,
        qa_id: str,
    ) -> bool:
        """
        Clear feedback for a QA entry (sets feedback_text and feedback_score to None).

        Returns True if updated, False if not found or cache unavailable.
        """
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
        user_id: str,
        session_id: str,
        qa_id: str,
    ) -> bool:
        """
        Delete a single QA entry by qa_id.

        Returns True if deleted, False if not found or cache unavailable.
        """
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_qa")
            return False

        return await self._cache.delete_qa_entry(
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
        )

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        Delete the entire session and all its QA entries.

        Returns True if deleted, False if session did not exist or cache unavailable.
        """
        if not self.is_available:
            logger.debug("SessionManager: cache unavailable, skipping delete_session")
            return False

        return await self._cache.delete_session(
            user_id=user_id,
            session_id=session_id,
        )


if __name__ == "__main__":
    import asyncio

    from cognee.infrastructure.session.get_session_manager import get_session_manager

    async def main():
        sm = get_session_manager()
        user_id, session_id = "test_user", "test_session"
        print("is_available:", sm.is_available)

        if not sm.is_available:
            print("Cache disabled, exiting")
            return

        qa_id1 = await sm.add_qa(user_id, session_id, "Q1?", "ctx1", "A1.")
        qa_id2 = await sm.add_qa(user_id, session_id, "Q2?", "ctx2", "A2.")
        print("added 2 entries:", qa_id1, qa_id2)

        entry = await sm.get_single_entry(user_id, session_id, qa_id1)
        print("get_single_entry(qa_id1):", entry)
        assert entry is not None and entry.get("question") == "Q1?"

        entry = await sm.get_single_entry(user_id, session_id, qa_id2)
        print("get_single_entry(qa_id2):", entry)
        assert entry is not None and entry.get("answer") == "A2."

        entry = await sm.get_single_entry(user_id, session_id, "nonexistent-id")
        print("get_single_entry(nonexistent):", entry)
        assert entry is None

        await sm.delete_session(user_id, session_id)
        print("get_single_entry tests OK.")

    asyncio.run(main())
