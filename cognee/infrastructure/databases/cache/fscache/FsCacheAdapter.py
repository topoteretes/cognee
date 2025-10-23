import logging
from redislite import Redis
from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface


class FSCacheAdapter(CacheDBInterface):
    def __init__(self, timeout: int, blocking_timeout: int, lock_key: str):
        self.redis_connection = Redis()
        self.timeout = timeout
        self.blocking_timeout = blocking_timeout
        self.lock_key = lock_key
        self.lock = None

    def acquire_lock(self):
        self.lock = self.redis_connection.lock(
            name=self.lock_key, timeout=self.timeout, blocking_timeout=self.blocking_timeout
        )
        acquired = self.lock.acquire()
        if not acquired:
            raise RuntimeError(f"Could not acquire lock: {self.lock_key}")

        return self.lock

    def release_lock(self):
        if self.lock is None:
            return
        self.lock.release()

    async def add_qa(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        ttl: int | None = 86400,
    ):
        raise NotImplementedError

    async def get_latest_qa(self, user_id: str, session_id: str, last_n: int = 5):
        raise NotImplementedError

    async def get_all_qas(self, user_id: str, session_id: str):
        raise NotImplementedError

    async def close(self):
        self.redis_connection.close()
