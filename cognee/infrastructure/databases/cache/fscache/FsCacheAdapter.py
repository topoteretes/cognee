import asyncio
import json
import os
from datetime import datetime
import time
import threading
import diskcache as dc

from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.exceptions.exceptions import (
    CacheConnectionError,
    SharedKuzuLockRequiresRedisError,
)
from cognee.infrastructure.files.storage.get_storage_config import get_storage_config
from cognee.shared.logging_utils import get_logger

logger = get_logger("FSCacheAdapter")


class FSCacheAdapter(CacheDBInterface):
    def __init__(self):
        default_key = "sessions_db"

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        cache_directory = os.path.join(data_root_directory, ".cognee_fs_cache", default_key)
        os.makedirs(cache_directory, exist_ok=True)
        self.cache = dc.Cache(directory=cache_directory)
        self.cache.expire()

        logger.debug(f"FSCacheAdapter initialized with cache directory: {cache_directory}")

    def acquire_lock(self):
        """Lock acquisition is not available for filesystem cache backend."""
        message = "Shared Kuzu lock requires Redis cache backend."
        logger.error(message)
        raise SharedKuzuLockRequiresRedisError()

    def release_lock(self):
        """Lock release is not available for filesystem cache backend."""
        message = "Shared Kuzu lock requires Redis cache backend."
        logger.error(message)
        raise SharedKuzuLockRequiresRedisError()

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
            return None
        entries = json.loads(value)
        return entries[-last_n:] if len(entries) > last_n else entries

    async def get_all_qas(self, user_id: str, session_id: str):
        session_key = f"agent_sessions:{user_id}:{session_id}"
        value = self.cache.get(session_key)
        if value is None:
            return None
        return json.loads(value)

    async def log_usage(
        self,
        user_id: str,
        log_entry: dict,
        ttl: int | None = 604800,
    ):
        """
        Usage logging is not supported in filesystem cache backend.
        This method is a no-op to satisfy the interface.
        """
        logger.warning("Usage logging not supported in FSCacheAdapter, skipping")
        pass

    async def get_usage_logs(self, user_id: str, limit: int = 100):
        """
        Usage logging is not supported in filesystem cache backend.
        This method returns an empty list to satisfy the interface.
        """
        logger.warning("Usage logging not supported in FSCacheAdapter, returning empty list")
        return []

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
