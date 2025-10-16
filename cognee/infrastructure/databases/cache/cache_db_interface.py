from abc import ABC, abstractmethod
from contextlib import contextmanager


class CacheDBInterface(ABC):
    """
    Abstract base class for distributed cache coordination systems (e.g., Redis, Memcached).
    Provides a common interface for lock acquisition, release, and context-managed locking.
    """

    def __init__(self, host: str, port: int, lock_key: str):
        self.host = host
        self.port = port
        self.lock_key = lock_key
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
