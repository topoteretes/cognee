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
        qa_id: Optional[str] = None,
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

        resolved_qa_id = qa_id or str(uuid.uuid4())
        await self._cache.create_qa_entry(
            user_id=user_id,
            session_id=session_id,
            question=question,
            context=context,
            answer=answer,
            qa_id=resolved_qa_id,
            feedback_text=feedback_text,
            feedback_score=feedback_score,
            ttl=ttl,
        )
        return resolved_qa_id

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

        qa_id1 = await sm.add_qa(user_id, session_id, "Q1?", "ctx1", "A1.", qa_id="id1")
        print("add_qa(qa_id=id1):", qa_id1)

        qa_id2 = await sm.add_qa(user_id, session_id, "Q2?", "ctx2", "A2.")
        print("add_qa(auto):", qa_id2)

        qa_id3 = await sm.add_qa(
            user_id, session_id, "Q3?", "ctx3", "A3.", feedback_text="good", feedback_score=4
        )
        print("add_qa(with feedback):", qa_id3)


        qa_id4 = await sm.add_qa('something_else', 'test_session', "Q4?", "ctx4", 'This is my answer')

        ok = await sm.delete_qa(user_id, session_id, qa_id2)


        ok = await sm.delete_session(user_id, session_id)

        not_ok = await sm.delete_qa(user_id, session_id, 'non_existent')

        not_ok = await sm.delete_session('non_exister', 'non_existent')


        print("All operations OK.")

    asyncio.run(main())
