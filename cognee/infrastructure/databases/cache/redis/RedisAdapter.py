import asyncio
import redis
import redis.asyncio as aioredis
from contextlib import contextmanager
from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.exceptions import CacheConnectionError
from cognee.shared.logging_utils import get_logger
from datetime import datetime
import json

logger = get_logger("RedisAdapter")


class RedisAdapter(CacheDBInterface):
    def __init__(
        self,
        host,
        port,
        lock_name="default_lock",
        username=None,
        password=None,
        timeout=240,
        blocking_timeout=300,
        connection_timeout=30,
    ):
        super().__init__(host, port, lock_name)

        self.host = host
        self.port = port
        self.connection_timeout = connection_timeout

        try:
            self.sync_redis = redis.Redis(
                host=host,
                port=port,
                username=username,
                password=password,
                socket_connect_timeout=connection_timeout,
                socket_timeout=connection_timeout,
            )
            self.async_redis = aioredis.Redis(
                host=host,
                port=port,
                username=username,
                password=password,
                decode_responses=True,
                socket_connect_timeout=connection_timeout,
            )
            self.timeout = timeout
            self.blocking_timeout = blocking_timeout

            # Validate connection on initialization
            self._validate_connection()
            logger.info(f"Successfully connected to Redis at {host}:{port}")

        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Failed to connect to Redis at {host}:{port}: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error initializing Redis adapter: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    def _validate_connection(self):
        """Validate Redis connection is available."""
        try:
            self.sync_redis.ping()
        except (redis.ConnectionError, redis.TimeoutError) as e:
            raise CacheConnectionError(
                f"Cannot connect to Redis at {self.host}:{self.port}: {str(e)}"
            ) from e

    def acquire_lock(self):
        """
        Acquire the Redis lock manually. Raises if acquisition fails. (Sync because of Kuzu)
        """
        self.lock = self.sync_redis.lock(
            name=self.lock_key,
            timeout=self.timeout,
            blocking_timeout=self.blocking_timeout,
        )

        acquired = self.lock.acquire()
        if not acquired:
            raise RuntimeError(f"Could not acquire Redis lock: {self.lock_key}")

        return self.lock

    def release_lock(self):
        """
        Release the Redis lock manually, if held. (Sync because of Kuzu)
        """
        if self.lock:
            try:
                self.lock.release()
                self.lock = None
            except redis.exceptions.LockError:
                pass

    @contextmanager
    def hold_lock(self):
        """
        Context manager for acquiring and releasing the Redis lock automatically. (Sync because of Kuzu)
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
        """
        Add a Q/A/context triplet to a Redis list for this session.
        Creates the session if it doesn't exist.

        Args:
            user_id (str): The user ID.
            session_id: Unique identifier for the session.
            question: User question text.
            context: Context used to answer.
            answer: Assistant answer text.
            ttl: Optional time-to-live (seconds). If provided, the session expires after this time.

        Raises:
            CacheConnectionError: If Redis connection fails or times out.
        """
        try:
            session_key = f"agent_sessions:{user_id}:{session_id}"

            qa_entry = {
                "time": datetime.utcnow().isoformat(),
                "question": question,
                "context": context,
                "answer": answer,
            }

            await self.async_redis.rpush(session_key, json.dumps(qa_entry))

            if ttl is not None:
                await self.async_redis.expire(session_key, ttl)

        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while adding Q&A: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while adding Q&A to Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def get_latest_qa(self, user_id: str, session_id: str, last_n: int = 5):
        """
        Retrieve the most recent Q/A/context triplet(s) for the given session.
        """
        session_key = f"agent_sessions:{user_id}:{session_id}"
        if last_n == 1:
            data = await self.async_redis.lindex(session_key, -1)
            return [json.loads(data)] if data else None
        else:
            data = await self.async_redis.lrange(session_key, -last_n, -1)
            return [json.loads(d) for d in data] if data else []

    async def get_all_qas(self, user_id: str, session_id: str):
        """
        Retrieve all Q/A/context triplets for the given session.
        """
        session_key = f"agent_sessions:{user_id}:{session_id}"
        entries = await self.async_redis.lrange(session_key, 0, -1)
        return [json.loads(e) for e in entries]

    async def close(self):
        """
        Gracefully close the async Redis connection.
        """
        await self.async_redis.aclose()


async def main():
    HOST = "localhost"
    PORT = 6379

    adapter = RedisAdapter(host=HOST, port=PORT)
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
