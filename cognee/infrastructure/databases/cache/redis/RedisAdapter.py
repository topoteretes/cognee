import json
import uuid
from contextlib import contextmanager
from datetime import datetime

import redis
import redis.asyncio as aioredis
from pydantic import BaseModel, ValidationError

from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.cache.models import SessionAgentTraceEntry, SessionQAEntry
from cognee.infrastructure.databases.exceptions import (
    CacheConnectionError,
    SessionQAEntryValidationError,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("RedisAdapter")


class RedisAdapter(CacheDBInterface):
    """Redis-backed cache adapter for session QA, trace storage, and coordination."""

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
        session_ttl_seconds: int | None = 604800,
    ):
        """Initialize sync/async Redis clients and validate connectivity up front."""
        super().__init__(host, port, lock_name, log_key)

        self.host = host
        self.port = port
        self.connection_timeout = connection_timeout
        self.session_ttl_seconds = session_ttl_seconds

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

    @staticmethod
    def _session_key(user_id: str, session_id: str) -> str:
        """Build the Redis key for QA session entries."""
        return f"agent_sessions:{user_id}:{session_id}"

    @staticmethod
    def _agent_trace_key(user_id: str, session_id: str) -> str:
        """Build the Redis key for agent trace entries."""
        return f"agent_traces:{user_id}:{session_id}"

    @staticmethod
    def _build_qa_entry_dump(
        question: str,
        context: str,
        answer: str,
        qa_id: str | None = None,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
        used_graph_element_ids: dict | None = None,
        memify_metadata: dict | None = None,
    ) -> dict:
        """Serialize one QA entry into the normalized Redis payload shape."""
        entry = SessionQAEntry(
            time=datetime.utcnow().isoformat(),
            question=question,
            context=context,
            answer=answer,
            qa_id=qa_id or str(uuid.uuid4()),
            feedback_text=feedback_text,
            feedback_score=feedback_score,
            used_graph_element_ids=used_graph_element_ids,
            memify_metadata=memify_metadata,
        )
        return entry.model_dump()

    @staticmethod
    def _build_agent_trace_entry_dump(
        trace_id: str,
        origin_function: str,
        status: str,
        memory_query: str = "",
        memory_context: str = "",
        method_params: dict | None = None,
        method_return_value=None,
        error_message: str = "",
        session_feedback: str = "",
    ) -> dict:
        """Serialize one agent-trace step into the normalized Redis payload shape."""
        entry = SessionAgentTraceEntry(
            trace_id=trace_id,
            origin_function=origin_function,
            status=status,
            memory_query=memory_query,
            memory_context=memory_context,
            method_params=method_params or {},
            method_return_value=method_return_value,
            error_message=error_message,
            session_feedback=session_feedback,
        )
        return entry.model_dump()

    async def _load_entries(self, session_key: str, start: int = 0, end: int = -1) -> list[dict]:
        """Load and deserialize a Redis list slice for the given key."""
        raw = await self.async_redis.lrange(session_key, start, end)
        return [json.loads(e) for e in raw] if raw else []

    async def _write_entry_at(self, session_key: str, index: int, entry_dump: dict) -> None:
        """Overwrite a single serialized entry in-place within a Redis list."""
        await self.async_redis.lset(session_key, index, json.dumps(entry_dump))

    async def _rewrite_entries(self, session_key: str, entries: list) -> None:
        """Replace the full Redis list contents for a session key."""
        await self.async_redis.delete(session_key)
        for entry in entries:
            await self.async_redis.rpush(session_key, json.dumps(entry))

    async def _apply_session_ttl(self, session_key: str) -> None:
        """Refresh the configured TTL for a session-scoped Redis key."""
        if self.session_ttl_seconds and self.session_ttl_seconds > 0:
            await self.async_redis.expire(session_key, self.session_ttl_seconds)

    @staticmethod
    def _merge_entry_update(
        entry: dict,
        question: str | None = None,
        context: str | None = None,
        answer: str | None = None,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
        used_graph_element_ids: dict | None = None,
        memify_metadata: dict | None = None,
    ) -> dict:
        """Merge partial QA updates into an existing serialized entry."""
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
        if used_graph_element_ids is not None:
            merged["used_graph_element_ids"] = used_graph_element_ids
        if memify_metadata is not None:
            existing_metadata = merged.get("memify_metadata")
            if isinstance(existing_metadata, dict):
                merged["memify_metadata"] = {**existing_metadata, **memify_metadata}
            else:
                merged["memify_metadata"] = memify_metadata
        return merged

    @staticmethod
    def _merge_entry_clear_feedback(entry: dict) -> dict:
        """Return a copy of the entry with feedback fields cleared."""
        return {**entry, "feedback_text": None, "feedback_score": None}

    @staticmethod
    def _validate_entry_dict(entry_dict: dict) -> dict:
        """Validate one serialized QA entry and return its normalized dump."""
        try:
            return SessionQAEntry.model_validate(entry_dict).model_dump()
        except ValidationError as e:
            raise SessionQAEntryValidationError(
                message=f"Session QA entry validation failed: {e!s}"
            ) from e

    @staticmethod
    def _find_index_by_qa_id(entries: list, qa_id: str) -> int | None:
        """Return the list index for a QA entry id, or None when absent."""
        for i, entry in enumerate(entries):
            if entry.get("qa_id") == qa_id:
                return i
        return None

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
        used_graph_element_ids: dict | None = None,
        memify_metadata: dict | None = None,
    ) -> None:
        """
        Add a Q/A/context triplet to a Redis list for this session.
        Same QA fields as update_qa_entry. Creates the session if it doesn't exist.
        """
        try:
            session_key = self._session_key(user_id, session_id)
            qa_entry = self._build_qa_entry_dump(
                question,
                context,
                answer,
                qa_id,
                feedback_text,
                feedback_score,
                used_graph_element_ids=used_graph_element_ids,
                memify_metadata=memify_metadata,
            )
            await self.async_redis.rpush(session_key, json.dumps(qa_entry))
            await self._apply_session_ttl(session_key)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while adding Q&A: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while adding Q&A to Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def get_latest_qa_entries(
        self, user_id: str, session_id: str, last_n: int = 5
    ) -> list[SessionQAEntry]:
        """
        Retrieve the most recent Q/A/context triplet(s) for the given session.
        """
        session_key = self._session_key(user_id, session_id)
        if last_n == 1:
            data = await self.async_redis.lindex(session_key, -1)
            return [SessionQAEntry.model_validate_json(data)] if data else None
        data = await self.async_redis.lrange(session_key, -last_n, -1)
        return [SessionQAEntry.model_validate_json(d) for d in data] if data else []

    async def get_all_qa_entries(self, user_id: str, session_id: str) -> list[SessionQAEntry]:
        """
        Retrieve all Q/A/context triplets for the given session.
        """
        session_key = self._session_key(user_id, session_id)
        return [SessionQAEntry(**entry) for entry in await self._load_entries(session_key)]

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
        used_graph_element_ids: dict | None = None,
        memify_metadata: dict | None = None,
    ) -> bool:
        """
        Update a QA entry by qa_id. Same QA fields as create_qa_entry.
        question/context/answer=None preserve existing values.
        Returns True if updated, False if qa_id not found.
        """
        try:
            session_key = self._session_key(user_id, session_id)
            entries = await self._load_entries(session_key)
            idx = self._find_index_by_qa_id(entries, qa_id)
            if idx is None:
                return False
            merged = self._merge_entry_update(
                entries[idx],
                question,
                context,
                answer,
                feedback_text,
                feedback_score,
                used_graph_element_ids=used_graph_element_ids,
                memify_metadata=memify_metadata,
            )
            entries[idx] = self._validate_entry_dict(merged)
            await self._write_entry_at(session_key, idx, entries[idx])
            await self._apply_session_ttl(session_key)
            return True
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
            session_key = self._session_key(user_id, session_id)
            entries = await self._load_entries(session_key)
            idx = self._find_index_by_qa_id(entries, qa_id)
            if idx is None:
                return False
            merged = self._merge_entry_clear_feedback(entries[idx])
            entries[idx] = self._validate_entry_dict(merged)
            await self._write_entry_at(session_key, idx, entries[idx])
            await self._apply_session_ttl(session_key)
            return True
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
            session_key = self._session_key(user_id, session_id)
            entries = await self._load_entries(session_key)
            idx = self._find_index_by_qa_id(entries, qa_id)
            if idx is None:
                return False
            entries.pop(idx)
            await self._rewrite_entries(session_key, entries)
            if entries:
                await self._apply_session_ttl(session_key)
            return True
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
        Delete the entire session and all its session-scoped artifacts.
        Returns True if any session data existed, False otherwise.
        """
        try:
            session_key = self._session_key(user_id, session_id)
            trace_key = self._agent_trace_key(user_id, session_id)
            deleted_sessions = await self.async_redis.delete(session_key)
            deleted_traces = await self.async_redis.delete(trace_key)
            return (deleted_sessions + deleted_traces) > 0

        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while deleting session: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while deleting session from Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def append_agent_trace_step(
        self,
        user_id: str,
        session_id: str,
        trace_id: str,
        origin_function: str,
        status: str,
        memory_query: str = "",
        memory_context: str = "",
        method_params: dict | None = None,
        method_return_value=None,
        error_message: str = "",
        session_feedback: str = "",
    ) -> None:
        """Append one trace step to the Redis list for this trace session."""
        try:
            trace_key = self._agent_trace_key(user_id, session_id)
            trace_entry = self._build_agent_trace_entry_dump(
                trace_id=trace_id,
                origin_function=origin_function,
                status=status,
                memory_query=memory_query,
                memory_context=memory_context,
                method_params=method_params,
                method_return_value=method_return_value,
                error_message=error_message,
                session_feedback=session_feedback,
            )
            await self.async_redis.rpush(trace_key, json.dumps(trace_entry))
            await self._apply_session_ttl(trace_key)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            error_msg = f"Redis connection error while appending agent trace step: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while appending agent trace step to Redis: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def get_agent_trace_session(
        self, user_id: str, session_id: str, last_n: int | None = None
    ) -> list[SessionAgentTraceEntry]:
        """Retrieve stored trace steps for the given session."""
        trace_key = self._agent_trace_key(user_id, session_id)
        if last_n is not None:
            return [
                SessionAgentTraceEntry(**entry)
                for entry in await self._load_entries(trace_key, -last_n, -1)
            ]
        return [SessionAgentTraceEntry(**entry) for entry in await self._load_entries(trace_key)]

    async def get_agent_trace_feedback(
        self, user_id: str, session_id: str, last_n: int | None = None
    ) -> list[str]:
        """Retrieve ordered per-step feedback for the given trace session."""
        entries = await self.get_agent_trace_session(user_id, session_id, last_n=last_n)
        return [entry.session_feedback for entry in entries]

    async def get_agent_trace_count(self, user_id: str, session_id: str) -> int:
        """Return the number of stored trace steps for the given session."""
        trace_key = self._agent_trace_key(user_id, session_id)
        return await self.async_redis.llen(trace_key)

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
