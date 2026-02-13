import json
import uuid
import os
from datetime import datetime
import diskcache as dc

from pydantic import ValidationError

from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.databases.exceptions.exceptions import (
    CacheConnectionError,
    SessionQAEntryValidationError,
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
        self.cache_directory = os.path.join(data_root_directory, ".cognee_fs_cache", default_key)
        os.makedirs(self.cache_directory, exist_ok=True)
        self.cache = dc.Cache(directory=self.cache_directory)
        self.cache.expire()

        logger.debug(f"FSCacheAdapter initialized with cache directory: {self.cache_directory}")

    @staticmethod
    def _session_key(user_id: str, session_id: str) -> str:
        return f"agent_sessions:{user_id}:{session_id}"

    @staticmethod
    def _build_qa_entry_dump(
        question: str,
        context: str,
        answer: str,
        qa_id: str | None = None,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
    ) -> dict:
        entry = SessionQAEntry(
            time=datetime.utcnow().isoformat(),
            question=question,
            context=context,
            answer=answer,
            qa_id=qa_id or str(uuid.uuid4()),
            feedback_text=feedback_text,
            feedback_score=feedback_score,
        )
        return entry.model_dump()

    def _load_entries(self, session_key: str) -> list:
        value = self.cache.get(session_key)
        if value is None:
            return []
        return json.loads(value)

    def _save_entries(self, session_key: str, entries: list) -> None:
        if entries:
            self.cache.set(session_key, json.dumps(entries))
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
    ) -> dict:
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
        return merged

    @staticmethod
    def _merge_entry_clear_feedback(entry: dict) -> dict:
        return {**entry, "feedback_text": None, "feedback_score": None}

    @staticmethod
    def _validate_entry_dict(entry_dict: dict) -> dict:
        try:
            return SessionQAEntry.model_validate(entry_dict).model_dump()
        except ValidationError as e:
            raise SessionQAEntryValidationError(
                message=f"Session QA entry validation failed: {e!s}"
            ) from e

    @staticmethod
    def _find_index_by_qa_id(entries: list, qa_id: str) -> int | None:
        for i, entry in enumerate(entries):
            if entry.get("qa_id") == qa_id:
                return i
        return None

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
        try:
            session_key = self._session_key(user_id, session_id)
            qa_entry = self._build_qa_entry_dump(
                question, context, answer, qa_id, feedback_text, feedback_score
            )
            entries = self._load_entries(session_key)
            entries.append(qa_entry)
            self._save_entries(session_key, entries)
        except Exception as e:
            error_msg = f"Unexpected error while adding Q&A to diskcache: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

    async def get_latest_qa_entries(self, user_id: str, session_id: str, last_n: int = 5):
        session_key = self._session_key(user_id, session_id)
        entries = self._load_entries(session_key)
        if not entries:
            return None
        return entries[-last_n:] if len(entries) > last_n else entries

    async def get_all_qa_entries(self, user_id: str, session_id: str):
        session_key = self._session_key(user_id, session_id)
        return self._load_entries(session_key)

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
        Only passed fields are updated; None preserves existing values.
        Returns True if updated, False if qa_id not found.
        """
        try:
            session_key = self._session_key(user_id, session_id)
            entries = self._load_entries(session_key)
            idx = self._find_index_by_qa_id(entries, qa_id)
            if idx is None:
                return False
            merged = self._merge_entry_update(
                entries[idx], question, context, answer, feedback_text, feedback_score
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
        Delete the entire session and all its QA entries.
        Returns True if deleted, False if session did not exist.
        """
        try:
            session_key = self._session_key(user_id, session_id)
            existed = self.cache.get(session_key) is not None
            if existed:
                self.cache.delete(session_key)
            return existed

        except Exception as e:
            error_msg = f"Unexpected error while deleting session from diskcache: {str(e)}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from e

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
        if self.cache is not None:
            self.cache.expire()
            self.cache.close()
