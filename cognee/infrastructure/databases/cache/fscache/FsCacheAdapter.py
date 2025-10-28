import asyncio
import json
import os
from datetime import datetime
import time
import threading
import diskcache as dc

from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.exceptions.exceptions import CacheConnectionError
from cognee.infrastructure.files.storage.get_storage_config import get_storage_config
from cognee.shared.logging_utils import get_logger

logger = get_logger("FSCacheAdapter")


class FSCacheAdapter(CacheDBInterface):
    def __init__(self, timeout=240, blocking_timeout=300, lock_key: str = "default_lock"):
        self.timeout = timeout
        self.blocking_timeout = blocking_timeout

        storage_config = get_storage_config()
        data_root_directory = storage_config.get["data_root_directory"]
        cache_directory = os.path.join(data_root_directory, ".cognee_fs_cache", lock_key)

        os.makedirs(cache_directory, exist_ok=True)

        self.cache = dc.Cache(directory=cache_directory)
        self.cache.expire()
        self.lock = dc.Lock(self.cache, lock_key)
        self._auto_release_timer = None
        self._timeout_flag = threading.Event()

        logger.debug(f"FSCacheAdapter initialized with cache directory: {cache_directory}")

    def acquire_lock(self):
        """
        Acquire the lock with timeout and auto-release settings.

        - timeout: How long to wait for lock acquisition before raising TimeoutError
        - blocking_timeout: Auto-release the lock after this duration (like TTL)

        Returns:
            bool: True if lock was acquired

        Raises:
            TimeoutError: If lock cannot be acquired within the timeout period
        """
        self._timeout_flag.clear()
        timeout_timer = None

        if self.timeout and self.timeout > 0:

            def on_timeout():
                self._timeout_flag.set()
                logger.error(
                    f"Failed to acquire lock within {self.timeout} seconds on key: {self.lock._key}"
                )

            timeout_timer = threading.Timer(self.timeout, on_timeout)
            timeout_timer.daemon = True
            timeout_timer.start()

        try:
            sleep_interval = 0.001

            while True:
                if self._timeout_flag.is_set():
                    raise TimeoutError(
                        f"Failed to acquire lock within {self.timeout} seconds on key: {self.lock._key}"
                    )

                if self.cache.add(
                    self.lock._key,
                    None,
                    expire=self.lock._expire,
                    tag=self.lock._tag,
                    retry=True,
                ):
                    logger.debug(f"Lock acquired successfully on key: {self.lock._key}")

                    if self.blocking_timeout and self.blocking_timeout > 0:

                        def on_auto_release():
                            logger.debug(
                                f"Auto-releasing lock after {self.blocking_timeout}s on key: {self.lock._key}"
                            )
                            self.release_lock()

                        self._auto_release_timer = threading.Timer(
                            self.blocking_timeout, on_auto_release
                        )
                        self._auto_release_timer.daemon = True
                        self._auto_release_timer.start()

                    return True

                time.sleep(sleep_interval)
        finally:
            if timeout_timer:
                timeout_timer.cancel()

    def release_lock(self):
        """Release the lock and cancel any auto-release timer."""
        if self.lock is None:
            return

        if self._auto_release_timer:
            self._auto_release_timer.cancel()
            logger.debug(f"Lock manually released on key: {self.lock._key}")

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
        try:
            session_key = f"agent_sessions:{user_id}:{session_id}"

            qa_entry = {
                "time": datetime.utcnow().isoformat(),
                "question": question,
                "context": context,
                "answer": answer,
            }

            existing_value = self.cache.get(session_key)
            if existing_value is not None:
                value: list = json.loads(existing_value)
                value.append(qa_entry)
            else:
                value = [qa_entry]

            self.cache.set(session_key, json.dumps(value), expire=ttl)
        except Exception as e:
            error_msg = f"Unexpected error while adding Q&A to diskcache: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def get_latest_qa(self, user_id: str, session_id: str, last_n: int = 5):
        session_key = f"agent_sessions:{user_id}:{session_id}"
        value = self.cache.get(session_key)
        if value is None:
            raise ValueError(f"key {session_key} does not exist in the cache")
        entries = json.loads(value)
        return entries[-last_n:] if len(entries) > last_n else entries

    async def get_all_qas(self, user_id: str, session_id: str):
        session_key = f"agent_sessions:{user_id}:{session_id}"
        value = self.cache.get(session_key)
        if value is None:
            return []
        return json.loads(value)

    async def close(self):
        if self.cache is not None:
            self.cache.expire()
            self.cache.close()


async def main():
    adapter = FSCacheAdapter()
    session_id = "demo_session"
    user_id = "demo_user_id"

    print("\nAdding sample Q/A pairs...")
    await adapter.add_qa(
        user_id,
        session_id,
        "What is Redis?",
        "Basic DB context",
        "Redis is an in-memory data store.",
    )
    await adapter.add_qa(
        user_id,
        session_id,
        "Who created Redis?",
        "Historical context",
        "Salvatore Sanfilippo (antirez).",
    )

    print("\nLatest QA:")
    latest = await adapter.get_latest_qa(user_id, session_id)
    print(json.dumps(latest, indent=2))

    print("\nLast 2 QAs:")
    last_two = await adapter.get_latest_qa(user_id, session_id, last_n=2)
    print(json.dumps(last_two, indent=2))

    session_id = "session_expire_demo"

    await adapter.add_qa(
        user_id,
        session_id,
        "What is Redis?",
        "Database context",
        "Redis is an in-memory data store.",
    )

    await adapter.add_qa(
        user_id,
        session_id,
        "Who created Redis?",
        "History context",
        "Salvatore Sanfilippo (antirez).",
    )

    print(await adapter.get_all_qas(user_id, session_id))

    await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
