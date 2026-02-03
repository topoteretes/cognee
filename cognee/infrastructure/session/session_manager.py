from typing import Optional

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
            default_session_id: Session ID to use when session_id is None.
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
