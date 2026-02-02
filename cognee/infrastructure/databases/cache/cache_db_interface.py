import uuid
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

    async def add_qa(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        ttl: int | None = 86400,
    ):
        """Backward-compatibility: delegates to create_qa_entry with generated qa_id. :TODO: delete when retrievers are updated"""
        return await self.create_qa_entry(
            user_id, session_id, question, context, answer,
            qa_id=str(uuid.uuid4()), ttl=ttl,
        )

    @abstractmethod
    async def create_qa_entry(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        qa_id: str,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
        ttl: int | None = 86400,
    ):
        """
        Add a Q/A/context triplet to a cache session.
        Uses the same QA fields as update_qa_entry for consistent structure.
        """
        pass

    async def get_latest_qa(self, user_id: str, session_id: str, last_n: int = 5):
        """Backward-compat: delegates to get_latest_qa_entries. :TODO: delete when retrievers are updated"""
        return await self.get_latest_qa_entries(user_id, session_id, last_n)

    @abstractmethod
    async def get_latest_qa_entries(self, user_id: str, session_id: str, last_n: int = 5):
        """
        Retrieve the most recent Q/A/context triplets for a session.
        """
        pass

    async def get_all_qas(self, user_id: str, session_id: str):
        """Backward-compat: delegates to get_all_qa_entries. :TODO: delete when retrievers are updated"""
        return await self.get_all_qa_entries(user_id, session_id)

    @abstractmethod
    async def get_all_qa_entries(self, user_id: str, session_id: str):
        """
        Retrieve all Q/A/context triplets for the given session.
        """
        pass

    @abstractmethod
    async def update_qa_entry(
        self,
        user_id: str,
        session_id: str,
        qa_id: str,
        question: str | None = None,
        context: str | None = None,
        answer: str | None = None,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
    ) -> bool:
        """
        Update a QA entry by qa_id. Same QA fields as create_qa_entry.
        Only passed fields are updated; None/default preserves existing values.
        Returns True if updated, False if qa_id not found.
        """
        pass

    @abstractmethod
    async def delete_qa_entries(self, user_id: str, session_id: str, qa_id: str) -> bool:
        """
        Delete a single QA entry by qa_id.
        Returns True if deleted, False if qa_id not found.
        """
        pass

    @abstractmethod
    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        Delete the entire session and all its QA entries.
        Returns True if deleted, False if session did not exist.
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
