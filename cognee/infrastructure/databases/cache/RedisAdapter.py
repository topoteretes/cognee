import redis
from contextlib import contextmanager


class RedisAdapter:
    def __init__(self, host, port, lock_name):
        self.redis = redis.Redis(host=host, port=port)
        self.lock_name = lock_name
        self.lock = None

    def acquire(self, timeout=240, blocking_timeout=300):
        """
        Acquire the Redis lock manually. Raises if acquisition fails.
        """
        self.lock = self.redis.lock(
            name=self.lock_name,
            timeout=timeout,
            blocking_timeout=blocking_timeout,
        )

        acquired = self.lock.acquire()
        if not acquired:
            raise RuntimeError(f"Could not acquire Redis lock: {self.lock_name}")

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
    def hold(self, timeout=60, blocking_timeout=300):
        """
        Context manager for acquiring and releasing the Redis lock automatically.
        """
        self.acquire(timeout=timeout, blocking_timeout=blocking_timeout)
        try:
            yield
        finally:
            self.release()
