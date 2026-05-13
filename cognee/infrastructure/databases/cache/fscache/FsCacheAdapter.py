import json
import os
import uuid
from datetime import datetime

import diskcache as dc
from pydantic import ValidationError

from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.cache.models import SessionAgentTraceEntry, SessionQAEntry
from cognee.infrastructure.databases.exceptions.exceptions import (
    CacheConnectionError,
    SessionQAEntryValidationError,
    SharedLadybugLockRequiresRedisError,
)
from cognee.infrastructure.files.storage.get_storage_config import get_storage_config
from cognee.shared.logging_utils import get_logger

logger = get_logger("FSCacheAdapter")


class FSCacheAdapter(CacheDBInterface):
    """Filesystem-backed cache adapter for session QA and agent-trace storage."""

    def __init__(self, session_ttl_seconds: int | None = 604800):
        """Initialize the disk-backed cache and eagerly evict expired entries."""
        default_key = "sessions_db"

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        self.cache_directory = os.path.join(data_root_directory, ".cognee_fs_cache", default_key)
        os.makedirs(self.cache_directory, exist_ok=True)
        self.cache = dc.Cache(directory=self.cache_directory)
        self.session_ttl_seconds = session_ttl_seconds
        # Evict any entries whose TTL has already elapsed
        self.cache.expire()

        logger.debug(f"FSCacheAdapter initialized with cache directory: {self.cache_directory}")

    @staticmethod
    def _session_key(user_id: str, session_id: str) -> str:
        """Build the storage key for QA session entries."""
        return f"agent_sessions:{user_id}:{session_id}"

    @staticmethod
    def _agent_trace_key(user_id: str, session_id: str) -> str:
        """Build the storage key for agent trace entries."""
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
        """Serialize one QA entry into the normalized cache payload shape."""
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
        """Serialize one agent-trace step into the normalized cache payload shape."""
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

    def _load_entries(self, session_key: str) -> list:
        """Load and deserialize all entries stored under the given cache key."""
        # Evict expired keys so stale sessions don't linger on disk
        self.cache.expire()
        value = self.cache.get(session_key)
        if value is None:
            return []
        return json.loads(value)

    def _save_entries(self, session_key: str, entries: list) -> None:
        """Persist the full entry list or delete the key when it becomes empty."""
        if entries:
            expire = (
                self.session_ttl_seconds
                if self.session_ttl_seconds and self.session_ttl_seconds > 0
                else None
            )
            self.cache.set(session_key, json.dumps(entries), expire=expire)
        else:
            self.cache.delete(session_key)

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
        """Merge partial QA updates into an existing entry payload."""
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
        """Lock acquisition is not available for filesystem cache backend."""
        message = "Shared Ladybug lock requires Redis cache backend."
        logger.error(message)
        raise SharedLadybugLockRequiresRedisError()

    def release_lock(self):
        """Lock release is not available for filesystem cache backend."""
        message = "Shared Ladybug lock requires Redis cache backend."
        logger.error(message)
        raise SharedLadybugLockRequiresRedisError()

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
        """Append one QA entry to the filesystem-backed session history."""
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
            with self.cache.transact():
                entries = self._load_entries(session_key)
                entries.append(qa_entry)
                self._save_entries(session_key, entries)
        except Exception as e:
            error_msg = f"Unexpected error while adding Q&A to diskcache: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def get_latest_qa_entries(
        self, user_id: str, session_id: str, last_n: int = 5
    ) -> list[SessionQAEntry]:
        """Return the most recent QA entries stored for the given session."""
        session_key = self._session_key(user_id, session_id)
        entries = [SessionQAEntry(**entry) for entry in self._load_entries(session_key)]
        if not entries:
            return []
        return entries[-last_n:] if len(entries) > last_n else entries

    async def get_all_qa_entries(self, user_id: str, session_id: str) -> list[SessionQAEntry]:
        """Return all QA entries stored for the given session."""
        session_key = self._session_key(user_id, session_id)
        return [SessionQAEntry(**entry) for entry in self._load_entries(session_key)]

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
        Only passed fields are updated; None preserves existing values.
        Returns True if updated, False if qa_id not found.
        """
        try:
            session_key = self._session_key(user_id, session_id)
            with self.cache.transact():
                entries = self._load_entries(session_key)
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
                self._save_entries(session_key, entries)
                return True
        except SessionQAEntryValidationError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error while updating Q&A in diskcache: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def delete_feedback(self, user_id: str, session_id: str, qa_id: str) -> bool:
        """
        Set feedback_text and feedback_score to None for a QA entry.
        """
        try:
            session_key = self._session_key(user_id, session_id)
            with self.cache.transact():
                entries = self._load_entries(session_key)
                idx = self._find_index_by_qa_id(entries, qa_id)
                if idx is None:
                    return False
                merged = self._merge_entry_clear_feedback(entries[idx])
                entries[idx] = self._validate_entry_dict(merged)
                self._save_entries(session_key, entries)
                return True
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
            with self.cache.transact():
                entries = self._load_entries(session_key)
                idx = self._find_index_by_qa_id(entries, qa_id)
                if idx is None:
                    return False
                entries.pop(idx)
                self._save_entries(session_key, entries)
                return True
        except Exception as e:
            error_msg = f"Unexpected error while deleting Q&A from diskcache: {str(e)}"
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
            qa_existed = self.cache.get(session_key) is not None
            trace_existed = self.cache.get(trace_key) is not None
            if qa_existed:
                self.cache.delete(session_key)
            if trace_existed:
                self.cache.delete(trace_key)
            return qa_existed or trace_existed

        except Exception as e:
            error_msg = f"Unexpected error while deleting session from diskcache: {str(e)}"
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
        """Append one trace step to the stored trace list for this session."""
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
            with self.cache.transact():
                entries = self._load_entries(trace_key)
                entries.append(trace_entry)
                self._save_entries(trace_key, entries)
        except Exception as e:
            error_msg = f"Unexpected error while appending agent trace step to diskcache: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def get_agent_trace_session(
        self, user_id: str, session_id: str, last_n: int | None = None
    ) -> list[SessionAgentTraceEntry]:
        """Retrieve stored trace steps for the given session."""
        trace_key = self._agent_trace_key(user_id, session_id)
        entries = [SessionAgentTraceEntry(**entry) for entry in self._load_entries(trace_key)]
        if last_n is not None:
            return entries[-last_n:]
        return entries

    async def get_agent_trace_feedback(
        self, user_id: str, session_id: str, last_n: int | None = None
    ) -> list[str]:
        """Retrieve ordered per-step feedback for the given trace session."""
        entries = await self.get_agent_trace_session(user_id, session_id, last_n=last_n)
        return [entry.session_feedback for entry in entries]

    async def get_agent_trace_count(self, user_id: str, session_id: str) -> int:
        """Return the number of stored trace steps for the given session."""
        trace_key = self._agent_trace_key(user_id, session_id)
        return len(self._load_entries(trace_key))

    async def prune(self) -> None:
        """
        Remove all items from the cache. In Cognee, prune means emptying the cache.
        Uses diskcache's clear() - does not delete the directory or recreate the cache.
        """
        try:
            self.cache.clear()
            self.cache.expire()

        except Exception as e:
            error_msg = f"Unexpected error while pruning diskcache: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

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
        """Flush diskcache expirations and close the underlying cache handle."""
        if self.cache is not None:
            self.cache.expire()
            self.cache.close()
