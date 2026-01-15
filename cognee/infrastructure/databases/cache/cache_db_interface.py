from abc import ABC, abstractmethod
from contextlib import contextmanager


class CacheDBInterface(ABC):
    """
    Abstract base class for distributed cache coordination systems (e.g., Redis, Memcached).
    Provides a common interface for lock acquisition, release, and context-managed locking.
    """

    def __init__(
        self, host: str, port: int, lock_key: str = "default_lock", log_key: str = "usage_logs"
    ):
        self.host = host
        self.port = port
        self.lock_key = lock_key
        self.log_key = log_key
        self.lock = None

    @abstractmethod
    def acquire_lock(self):
        """
        Acquire a lock on the given key.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def release_lock(self):
        """
        Release the lock if it is held.
        Must be implemented by subclasses.
        """
        pass

    @contextmanager
    def hold_lock(self):
        """
        Context manager for safely acquiring and releasing the lock.
        """
        self.acquire()
        try:
            yield
        finally:
            self.release()

    @abstractmethod
    async def add_qa(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        ttl: int | None = 86400,
    ):
        """
        Add a Q/A/context triplet to a cache session.
        """

        pass

    @abstractmethod
    async def get_latest_qa(self, user_id: str, session_id: str, last_n: int = 5):
        """
        Retrieve the most recent Q/A/context triplets for a session.
        """
        pass

    @abstractmethod
    async def get_all_qas(self, user_id: str, session_id: str):
        """
        Retrieve all Q/A/context triplets for the given session.
        """
        pass

    @abstractmethod
    async def close(self):
        """
        Gracefully close any async connections.
        """
        pass

    @abstractmethod
    async def log_usage(
        self,
        user_id: str,
        log_entry: dict,
        ttl: int | None = 604800,
    ):
        """
        Log usage information (API endpoint calls, MCP tool invocations) to cache.

        Args:
            user_id: The user ID.
            log_entry: Dictionary containing usage log information.
            ttl: Optional time-to-live (seconds). If provided, the log list expires after this time.

        Raises:
            CacheConnectionError: If cache connection fails or times out.
        """
        pass

    @abstractmethod
    async def get_usage_logs(self, user_id: str, limit: int = 100):
        """
        Retrieve usage logs for a given user.

        Args:
            user_id: The user ID.
            limit: Maximum number of logs to retrieve (default: 100).

        Returns:
            List of usage log entries, most recent first.
        """
        pass
