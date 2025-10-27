import asyncio
import json
from datetime import datetime
import fakeredis

from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.exceptions.exceptions import CacheConnectionError
from cognee.shared.logging_utils import get_logger

logger = get_logger("FSCacheAdapter")


class FSCacheAdapter(CacheDBInterface):
    def __init__(self, timeout=240, blocking_timeout=300, lock_key: str = "default_lock"):
        self.redis_connection = fakeredis.FakeStrictRedis(server_type="redis")
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
        try:
            session_key = f"agent_sessions:{user_id}:{session_id}"

            qa_entry = {
                "time": datetime.utcnow().isoformat(),
                "question": question,
                "context": context,
                "answer": answer,
            }

            self.redis_connection.rpush(session_key, json.dumps(qa_entry))

            if ttl is not None:
                self.redis_connection.expire(session_key, ttl)
        except Exception as e:
            error_msg = f"Unexpected error while adding Q&A to RedisLite: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def get_latest_qa(self, user_id: str, session_id: str, last_n: int = 5):
        session_key = f"agent_sessions:{user_id}:{session_id}"
        if last_n == 1:
            data = self.redis_connection.lindex(session_key, -1)
            return [json.loads(data)] if data else None
        else:
            data = self.redis_connection.lrange(session_key, -last_n, -1)
            return [json.loads(d) for d in data] if data else []

    async def get_all_qas(self, user_id: str, session_id: str):
        session_key = f"agent_sessions:{user_id}:{session_id}"
        entries = self.redis_connection.lrange(session_key, 0, -1)
        return [json.loads(e) for e in entries]

    async def close(self):
        self.redis_connection.close()


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
