"""
Regression tests for CacheDBInterface.hold_lock().

Covers three scenarios mandated by issue #3294:
  (a) acquire_lock returns None  -> release_lock is NOT called.
  (b) acquire_lock returns a valid (truthy) handle -> release_lock IS called
      with that exact handle.
  (c) acquire_lock raises        -> release_lock is NOT called.
"""
import pytest
from unittest.mock import MagicMock

from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface


# ---------------------------------------------------------------------------
# Minimal concrete subclass – only implements the two lock methods so the ABC
# check is satisfied without pulling in any real infrastructure.
# ---------------------------------------------------------------------------

class _MinimalAdapter(CacheDBInterface):
    """Concrete stub that delegates lock calls to injected mocks."""

    def __init__(self, acquire_mock, release_mock):
        # Skip the real __init__ (needs host/port) – directly set attributes.
        self.host = "localhost"
        self.port = 6379
        self.lock_key = "test_lock"
        self.log_key = "test_logs"
        self.lock = None
        self._acquire_mock = acquire_mock
        self._release_mock = release_mock

    # --- Required abstract methods (lock pair) ---
    def acquire_lock(self):
        return self._acquire_mock()

    def release_lock(self, lock=None):
        return self._release_mock(lock)

    # --- Remaining abstract methods – stubs to satisfy ABC ---
    async def create_qa_entry(self, user_id, session_id, question, context, answer, qa_id,
                              feedback_text=None, feedback_score=None,
                              used_graph_element_ids=None, memify_metadata=None,
                              used_session_context_ids=None):
        raise NotImplementedError("stub")

    async def get_latest_qa_entries(self, user_id, session_id, last_n=5):
        raise NotImplementedError("stub")

    async def get_all_qa_entries(self, user_id, session_id):
        raise NotImplementedError("stub")

    async def get_qa_entries_by_ids(self, user_id, session_id, qa_ids):
        raise NotImplementedError("stub")

    async def update_qa_entry(self, user_id, session_id, qa_id, question=None, context=None,
                              answer=None, feedback_text=None, feedback_score=None,
                              used_graph_element_ids=None, memify_metadata=None,
                              used_session_context_ids=None):
        raise NotImplementedError("stub")

    async def delete_feedback(self, user_id, session_id, qa_id):
        raise NotImplementedError("stub")

    async def delete_qa_entry(self, user_id, session_id, qa_id):
        raise NotImplementedError("stub")

    async def delete_session(self, user_id, session_id):
        raise NotImplementedError("stub")

    async def append_agent_trace_step(self, user_id, session_id, trace_id, origin_function,
                                      status, memory_query="", memory_context="",
                                      method_params=None, method_return_value=None,
                                      error_message="", session_feedback=""):
        raise NotImplementedError("stub")

    async def get_agent_trace_session(self, user_id, session_id, last_n=None):
        raise NotImplementedError("stub")

    async def get_agent_trace_feedback(self, user_id, session_id, last_n=None):
        raise NotImplementedError("stub")

    async def get_agent_trace_count(self, user_id, session_id):
        raise NotImplementedError("stub")

    async def create_session_context_entry(self, user_id, session_id, entry_dump):
        raise NotImplementedError("stub")

    async def get_session_context_entries(self, user_id, session_id):
        raise NotImplementedError("stub")

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        raise NotImplementedError("stub")

    async def delete_session_context(self, user_id, session_id):
        raise NotImplementedError("stub")

    async def prune(self):
        raise NotImplementedError("stub")

    async def close(self):
        raise NotImplementedError("stub")

    async def log_usage(self, user_id, log_entry, ttl=604800):
        raise NotImplementedError("stub")

    async def get_usage_logs(self, user_id, limit=100):
        raise NotImplementedError("stub")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(acquire_return=None, acquire_raises=None):
    acquire_mock = MagicMock()
    release_mock = MagicMock()
    if acquire_raises is not None:
        acquire_mock.side_effect = acquire_raises
    else:
        acquire_mock.return_value = acquire_return
    adapter = _MinimalAdapter(acquire_mock, release_mock)
    return adapter, acquire_mock, release_mock


# ---------------------------------------------------------------------------
# Scenario (a): acquire_lock returns None -> release_lock NOT called
# ---------------------------------------------------------------------------

class TestHoldLockAcquireReturnsNone:
    """When acquire_lock() returns None, release_lock must never be invoked."""

    def test_release_lock_not_called_when_acquire_returns_none(self):
        adapter, acquire_mock, release_mock = _make_adapter(acquire_return=None)

        with adapter.hold_lock():
            pass  # body executes normally

        acquire_mock.assert_called_once()
        release_mock.assert_not_called()

    def test_yielded_value_is_none_when_acquire_returns_none(self):
        adapter, _, _ = _make_adapter(acquire_return=None)

        with adapter.hold_lock() as lock:
            assert lock is None

    def test_release_lock_not_called_even_if_body_raises(self):
        adapter, acquire_mock, release_mock = _make_adapter(acquire_return=None)

        with pytest.raises(RuntimeError):
            with adapter.hold_lock():
                raise RuntimeError("body error")

        release_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Scenario (b): acquire_lock returns a valid handle -> release_lock IS called
# ---------------------------------------------------------------------------

class TestHoldLockAcquireReturnsValidHandle:
    """When acquire_lock() returns a truthy handle, release_lock must be called with it."""

    def test_release_lock_called_with_correct_handle(self):
        sentinel = object()
        adapter, acquire_mock, release_mock = _make_adapter(acquire_return=sentinel)

        with adapter.hold_lock():
            pass

        release_mock.assert_called_once_with(sentinel)

    def test_yielded_value_is_the_lock_handle(self):
        sentinel = object()
        adapter, _, _ = _make_adapter(acquire_return=sentinel)

        with adapter.hold_lock() as lock:
            assert lock is sentinel

    def test_release_lock_called_even_if_body_raises(self):
        """release_lock must run in the finally block regardless of body exceptions."""
        sentinel = object()
        adapter, acquire_mock, release_mock = _make_adapter(acquire_return=sentinel)

        with pytest.raises(ValueError):
            with adapter.hold_lock():
                raise ValueError("body error")

        release_mock.assert_called_once_with(sentinel)

    def test_release_lock_called_with_string_handle(self):
        adapter, _, release_mock = _make_adapter(acquire_return="lock-id-abc")

        with adapter.hold_lock():
            pass

        release_mock.assert_called_once_with("lock-id-abc")

    def test_release_lock_called_with_integer_handle(self):
        adapter, _, release_mock = _make_adapter(acquire_return=42)

        with adapter.hold_lock():
            pass

        release_mock.assert_called_once_with(42)


# ---------------------------------------------------------------------------
# Scenario (c): acquire_lock raises -> release_lock NOT called
# ---------------------------------------------------------------------------

class TestHoldLockAcquireRaises:
    """When acquire_lock() raises, release_lock must never be invoked."""

    def test_release_lock_not_called_when_acquire_raises(self):
        adapter, acquire_mock, release_mock = _make_adapter(
            acquire_raises=RuntimeError("lock acquisition failed")
        )

        with pytest.raises(RuntimeError, match="lock acquisition failed"):
            with adapter.hold_lock():
                pass  # should never reach here

        release_mock.assert_not_called()

    def test_release_lock_not_called_on_timeout_error(self):
        adapter, acquire_mock, release_mock = _make_adapter(
            acquire_raises=TimeoutError("timed out waiting for lock")
        )

        with pytest.raises(TimeoutError):
            with adapter.hold_lock():
                pass

        release_mock.assert_not_called()

    def test_acquire_exception_propagates_to_caller(self):
        exc = Exception("unexpected error during acquire")
        adapter, acquire_mock, release_mock = _make_adapter(acquire_raises=exc)

        with pytest.raises(Exception, match="unexpected error during acquire"):
            with adapter.hold_lock():
                pass

        release_mock.assert_not_called()
