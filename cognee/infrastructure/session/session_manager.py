import uuid
from typing import Optional, Union

from cognee.shared.logging_utils import get_logger

logger = get_logger("SessionManager")


class SessionManager:
    """
    Manages session QA entries. Wraps the cache engine with a domain-focused API.
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
            question=question,
            context=context,
            answer=answer,
            qa_id=qa_id,
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

    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    async def main():
        config = get_cache_config()
        backend = config.cache_backend
        print(f"Cache backend: {backend} (set CACHE_BACKEND=redis or fs to switch)")

        sm = get_session_manager()
        user_id, session_id = "test_user", "test_session"
        print("is_available:", sm.is_available)

        if not sm.is_available:
            print("Cache disabled, exiting")
            return

        sid = sm.normalize_session_id(None)
        assert sid == "default_session"
        print("normalize_session_id(None):", sid)

        qa_id1 = await sm.add_qa(user_id, session_id, "Q1?", "ctx1", "A1.")
        print("add_qa(qa_id=id1):", qa_id1)

        qa_id2 = await sm.add_qa(user_id, session_id, "Q2?", "ctx2", "A2.")
        print("add_qa(auto):", qa_id2)

        qa_id3 = await sm.add_qa(
            user_id, session_id, "Q3?", "ctx3", "A3.", feedback_text="good", feedback_score=4
        )
        print("add_qa(with feedback):", qa_id3)

        entries = await sm.get_session(user_id, session_id)
        print("get_session (all):", len(entries), "entries")

        last2 = await sm.get_session(user_id, session_id, last_n=2)
        print("get_session(last_n=2):", len(last2), "entries")

        formatted = await sm.get_session(user_id, session_id, last_n=2, formatted=True)
        print("get_session(formatted=True) len:", len(formatted))
        assert "Previous conversation" in formatted and "Q2?" in formatted

        raw = await sm.get_session(user_id, session_id, last_n=2, formatted=False)
        print("get_session(formatted=False):", len(raw), "entries")
        assert isinstance(raw, list) and len(raw) == 2

        ok = await sm.update_qa(user_id, session_id, qa_id1, question="Q1 updated?", answer="A1 updated.")
        print("update_qa(id1):", ok)
        entries = await sm.get_session(user_id, session_id)
        e1 = next(e for e in entries if e["qa_id"] == qa_id1)
        assert e1["question"] == "Q1 updated?"

        entries = await sm.get_session(user_id, session_id)
        e1 = next(e for e in entries if e["qa_id"] == qa_id1)
        assert e1.get("feedback_score") is None and e1.get("feedback_text") is None

        ok = await sm.delete_qa(user_id, session_id, qa_id2)
        print("delete_qa(qa_id2):", ok)
        entries = await sm.get_session(user_id, session_id)
        print("after delete_qa:", len(entries), "entries")

        ok = await sm.delete_session(user_id, session_id)
        print("delete_session:", ok)
        entries = await sm.get_session(user_id, session_id)
        assert len(entries) == 0
        print("after delete_session:", len(entries), "entries")

        fmt = sm.format_entries([])
        assert fmt == ""
        print("format_entries([]): empty")

        print()

    asyncio.run(main())
