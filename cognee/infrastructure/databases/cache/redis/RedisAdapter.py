import uuid
import redis
import redis.asyncio as aioredis
from contextlib import contextmanager
from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from pydantic import ValidationError

from cognee.infrastructure.databases.exceptions import (
    CacheConnectionError,
    SessionQAEntryValidationError,
)
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
        log_key="usage_logs",
        username=None,
        password=None,
        timeout=240,
        blocking_timeout=300,
        connection_timeout=30,
    ):
        super().__init__(host, port, lock_name, log_key)

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

    async def create_qa_entry(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        qa_id: str | None = None,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
    ):
        """
        Add a Q/A/context triplet to a Redis list for this session.
        Same QA fields as update_qa_entry. Creates the session if it doesn't exist.
        """
        try:
            session_key = f"agent_sessions:{user_id}:{session_id}"

            entry = SessionQAEntry(
                time=datetime.utcnow().isoformat(),
                question=question,
                context=context,
                answer=answer,
                qa_id=qa_id or str(uuid.uuid4()),
                feedback_text=feedback_text,
                feedback_score=feedback_score,
            )
            await self.async_redis.rpush(session_key, json.dumps(entry.model_dump()))

        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while adding Q&A: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while adding Q&A to Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def get_latest_qa_entries(self, user_id: str, session_id: str, last_n: int = 5):
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

    async def get_all_qa_entries(self, user_id: str, session_id: str):
        """
        Retrieve all Q/A/context triplets for the given session.
        """
        session_key = f"agent_sessions:{user_id}:{session_id}"
        entries = await self.async_redis.lrange(session_key, 0, -1)
        return [json.loads(e) for e in entries]

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
        question/context/answer=None preserve existing values.
        Returns True if updated, False if qa_id not found.
        """
        try:
            session_key = f"agent_sessions:{user_id}:{session_id}"
            entries_raw = await self.async_redis.lrange(session_key, 0, -1)
            if not entries_raw:
                return False

            entries = [json.loads(e) for e in entries_raw]
            for i, entry in enumerate(entries):
                if entry.get("qa_id") == qa_id:
                    merged = {**entry}
                    if question is not None:
                        merged["question"] = question
                    if context is not None:
                        merged["context"] = context
                    if answer is not None:
                        merged["answer"] = answer
                    if feedback_text is not None:
                        merged["feedback_text"] = feedback_text
                    if feedback_score is not None:
                        merged["feedback_score"] = feedback_score
                    try:
                        validated = SessionQAEntry.model_validate(merged)
                    except ValidationError as e:
                        raise SessionQAEntryValidationError(
                            message=f"Session QA entry validation failed during update_qa_entry operation: {e!s}"
                        ) from e
                    await self.async_redis.lset(session_key, i, json.dumps(validated.model_dump()))
                    return True
            return False

        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while updating Q&A: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except SessionQAEntryValidationError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error while updating Q&A in Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def delete_feedback(self, user_id: str, session_id: str, qa_id: str) -> bool:
        """
        Set feedback_text and feedback_score to None for a QA entry.
        """
        try:
            session_key = f"agent_sessions:{user_id}:{session_id}"
            entries_raw = await self.async_redis.lrange(session_key, 0, -1)
            if not entries_raw:
                return False

            entries = [json.loads(e) for e in entries_raw]
            for i, entry in enumerate(entries):
                if entry.get("qa_id") == qa_id:
                    merged = {**entry, "feedback_text": None, "feedback_score": None}
                    try:
                        validated = SessionQAEntry.model_validate(merged)
                    except ValidationError as e:
                        raise SessionQAEntryValidationError(
                            message=f"Session QA entry validation failed: {e!s}"
                        ) from e
                    await self.async_redis.lset(session_key, i, json.dumps(validated.model_dump()))
                    return True
            return False

        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while clearing feedback: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except SessionQAEntryValidationError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error while clearing feedback: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def delete_qa_entry(self, user_id: str, session_id: str, qa_id: str) -> bool:
        """
        Delete a single QA entry by qa_id.
        Returns True if deleted, False if qa_id not found.
        """
        try:
            session_key = f"agent_sessions:{user_id}:{session_id}"
            entries_raw = await self.async_redis.lrange(session_key, 0, -1)
            if not entries_raw:
                return False

            entries = [json.loads(e) for e in entries_raw]
            for i, entry in enumerate(entries):
                if entry.get("qa_id") == qa_id:
                    entries.pop(i)
                    await self.async_redis.delete(session_key)
                    for e in entries:
                        await self.async_redis.rpush(session_key, json.dumps(e))
                    return True
            return False

        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while deleting Q&A: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while deleting Q&A from Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        Delete the entire session and all its QA entries.
        Returns True if deleted, False if session did not exist.
        """
        try:
            session_key = f"agent_sessions:{user_id}:{session_id}"
            deleted = await self.async_redis.delete(session_key)
            return deleted > 0

        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while deleting session: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while deleting session from Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def prune(self) -> None:
        """
        Flush the Redis database. In Cognee, prune means deleting the whole cache.
        """
        try:
            await self.async_redis.flushdb()

        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while pruning: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while pruning Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def log_usage(
        self,
        user_id: str,
        log_entry: dict,
        ttl: int | None = 604800,
    ):
        """
        Log usage information (API endpoint calls, MCP tool invocations) to Redis.

        Args:
            user_id: The user ID.
            log_entry: Dictionary containing usage log information.
            ttl: Optional time-to-live (seconds). If provided, the log list expires after this time.

        Raises:
            CacheConnectionError: If Redis connection fails or times out.
        """
        try:
            usage_logs_key = f"{self.log_key}:{user_id}"

            await self.async_redis.rpush(usage_logs_key, json.dumps(log_entry))

            if ttl is not None:
                await self.async_redis.expire(usage_logs_key, ttl)

        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while logging usage: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while logging usage to Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def get_usage_logs(self, user_id: str, limit: int = 100):
        """
        Retrieve usage logs for a given user.

        Args:
            user_id: The user ID.
            limit: Maximum number of logs to retrieve (default: 100).

        Returns:
            List of usage log entries, most recent first.
        """
        try:
            usage_logs_key = f"{self.log_key}:{user_id}"
            entries = await self.async_redis.lrange(usage_logs_key, -limit, -1)
            return [json.loads(e) for e in reversed(entries)] if entries else []
        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while retrieving usage logs: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while retrieving usage logs from Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def close(self):
        """Close Redis connections."""
        try:
            await self.async_redis.aclose()
        except Exception as e:
            logger.debug("Error closing Redis async connection: %s", e)
