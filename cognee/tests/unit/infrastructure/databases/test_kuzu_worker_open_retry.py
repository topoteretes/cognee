"""Tests for the worker-side lock-retry on opening a Ladybug database
(``cognee_db_workers.kuzu_worker._retry_open_locked``).

These don't import ``ladybug`` — the worker imports it lazily and the retry
helper is driven with a fake stand-in.
"""

from __future__ import annotations

import pytest

import cognee_db_workers.harness as harness
from cognee_db_workers import kuzu_worker

_LOCK_MSG = "IO exception: Could not set lock on file : /tmp/x"


class _FakeLadybug:
    """Stand-in for the ``ladybug`` module. ``Database(**kwargs)`` raises a
    lock-held ``RuntimeError`` the first ``fail_times`` calls, then succeeds."""

    def __init__(self, fail_times, error_message=_LOCK_MSG):
        self._left = fail_times
        self._error_message = error_message
        self.calls = 0

    def Database(self, **kwargs):
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise RuntimeError(self._error_message)
        return "DB_OK"


def test_retry_disabled_reraises_original_not_typeerror(monkeypatch):
    """SUBPROCESS_OPEN_LOCK_RETRIES <= 0 must re-raise the original lock error,
    not blow up with ``TypeError: exceptions must derive from BaseException``
    from ``raise None``."""
    monkeypatch.setattr(harness, "OPEN_LOCK_RETRIES", 0)
    monkeypatch.setattr(harness, "OPEN_LOCK_BACKOFF", 0.001)

    fake = _FakeLadybug(fail_times=99)  # would keep failing if ever called
    original = RuntimeError(_LOCK_MSG)

    with pytest.raises(RuntimeError) as exc_info:
        kuzu_worker._retry_open_locked(fake, {"database_path": "/tmp/x"}, original)

    assert exc_info.value is original
    assert fake.calls == 0, "retries disabled must not attempt any reopen"


def test_retry_succeeds_after_transient_lock(monkeypatch):
    """A lock that clears within the retry budget yields a successful open."""
    monkeypatch.setattr(harness, "OPEN_LOCK_RETRIES", 5)
    monkeypatch.setattr(harness, "OPEN_LOCK_BACKOFF", 0.001)

    fake = _FakeLadybug(fail_times=2)
    result = kuzu_worker._retry_open_locked(fake, {}, RuntimeError(_LOCK_MSG))

    assert result == "DB_OK"
    assert fake.calls == 3  # 2 transient failures + 1 success


def test_retry_exhausted_reraises_last_lock_error(monkeypatch):
    """When every retry still hits the lock, the last lock error propagates."""
    monkeypatch.setattr(harness, "OPEN_LOCK_RETRIES", 3)
    monkeypatch.setattr(harness, "OPEN_LOCK_BACKOFF", 0.001)

    fake = _FakeLadybug(fail_times=99)
    with pytest.raises(RuntimeError, match="Could not set lock on file"):
        kuzu_worker._retry_open_locked(fake, {}, RuntimeError(_LOCK_MSG))
    assert fake.calls == 3


def test_retry_passes_through_non_lock_error(monkeypatch):
    """A non-lock RuntimeError during a retry attempt is not swallowed/retried."""
    monkeypatch.setattr(harness, "OPEN_LOCK_RETRIES", 5)
    monkeypatch.setattr(harness, "OPEN_LOCK_BACKOFF", 0.001)

    fake = _FakeLadybug(fail_times=99, error_message="some other corruption")
    with pytest.raises(RuntimeError, match="some other corruption"):
        kuzu_worker._retry_open_locked(fake, {}, RuntimeError(_LOCK_MSG))
    assert fake.calls == 1, "a non-lock error must abort retries immediately"
