import redis
from contextlib import contextmanager
from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface


class RedisAdapter(CacheDBInterface):
    def __init__(self, host, port, lock_name, timeout=240, blocking_timeout=300):
        super().__init__(host, port, lock_name)
        self.redis = redis.Redis(host=host, port=port)
        self.timeout = timeout
        self.blocking_timeout = blocking_timeout

    def acquire(self):
        """
        Acquire the Redis lock manually. Raises if acquisition fails.
        """
        self.lock = self.redis.lock(
            name=self.lock_key,
            timeout=self.timeout,
            blocking_timeout=self.blocking_timeout,
        )

        acquired = self.lock.acquire()
        if not acquired:
            raise RuntimeError(f"Could not acquire Redis lock: {self.lock_key}")

        return self.lock

    def release(self):
        """
        Release the Redis lock manually, if held.
        """
        if self.lock:
            try:
                self.lock.release()
                self.lock = None
            except redis.exceptions.LockError:
                pass

    @contextmanager
    def hold(self):
        """
        Context manager for acquiring and releasing the Redis lock automatically.
        """
        self.acquire()
        try:
            yield
        finally:
            self.release()
