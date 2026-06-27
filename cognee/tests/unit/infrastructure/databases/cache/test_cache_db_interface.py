import pytest
from unittest.mock import MagicMock
from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface


class MockCacheDB(CacheDBInterface):
    def acquire_lock(self):
        pass

    def release_lock(self, lock=None):
        pass

    async def create_qa_entry(self, *args, **kwargs):
        pass

    async def get_latest_qa_entries(self, *args, **kwargs):
        pass

    async def get_all_qa_entries(self, *args, **kwargs):
        pass

    async def get_qa_entries_by_ids(self, *args, **kwargs):
        pass

    async def update_qa_entry(self, *args, **kwargs):
        pass

    async def delete_feedback(self, *args, **kwargs):
        pass

    async def delete_qa_entry(self, *args, **kwargs):
        pass

    async def delete_session(self, *args, **kwargs):
        pass

    async def append_agent_trace_step(self, *args, **kwargs):
        pass

    async def get_agent_trace_session(self, *args, **kwargs):
        pass

    async def get_agent_trace_feedback(self, *args, **kwargs):
        pass

    async def get_agent_trace_count(self, *args, **kwargs):
        pass

    async def create_session_context_entry(self, *args, **kwargs):
        pass

    async def get_session_context_entries(self, *args, **kwargs):
        pass

    async def update_session_context_entry(self, *args, **kwargs):
        pass

    async def delete_session_context(self, *args, **kwargs):
        pass

    async def prune(self, *args, **kwargs):
        pass

    async def close(self, *args, **kwargs):
        pass

    async def log_usage(self, *args, **kwargs):
        pass

    async def get_usage_logs(self, *args, **kwargs):
        pass


def test_hold_lock_bypasses_release_on_failed_acquisition():
    cache_db = MockCacheDB(host="localhost", port=6379)
    cache_db.acquire_lock = MagicMock(return_value=None)
    cache_db.release_lock = MagicMock()
    with cache_db.hold_lock():
        pass
    cache_db.acquire_lock.assert_called_once()
    cache_db.release_lock.assert_not_called()


def test_hold_lock_releases_on_successful_acquisition():
    cache_db = MockCacheDB(host="localhost", port=6379)
    mock_lock_handle = "mock_redis_lock_uuid_1234"
    cache_db.acquire_lock = MagicMock(return_value=mock_lock_handle)
    cache_db.release_lock = MagicMock()
    with cache_db.hold_lock():
        pass
    cache_db.acquire_lock.assert_called_once()
    cache_db.release_lock.assert_called_once_with(mock_lock_handle)
