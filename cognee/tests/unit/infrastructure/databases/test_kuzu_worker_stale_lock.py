"""Tests for stale-lock reclamation when opening a Ladybug database
(``cognee_db_workers.kuzu_worker._reclaim_stale_lock`` and
``_open_locked_with_recovery``).

A worker that crashes can leave ``<db_path>.lock`` on disk, after which every
open fails with "Lock is held by PID <pid>" for a PID that no longer exists,
wedging all later recall() calls (gh #3708). These tests pin the recovery: the
stale lock is removed only when its holder PID is confirmed dead, and a live or
unknown holder is left untouched. They don't import ``ladybug`` — the worker
imports it lazily and the helpers are driven with fakes.
"""

from __future__ import annotations

import os

import pytest

import cognee_db_workers.harness as harness
from cognee_db_workers import kuzu_worker

_LOCK_MSG_TMPL = "IO exception: Could not set lock on file : {path} (Lock is held by PID {pid})"


class _FakeLadybug:
    """Stand-in for the ``ladybug`` module whose ``Database(**kwargs)`` raises a
    lock-held ``RuntimeError`` ``fail_times`` times, then returns ``"DB_OK"``."""

    def __init__(self, fail_times, error_message):
        self._left = fail_times
        self._error_message = error_message
        self.calls = 0

    def Database(self, **kwargs):
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise RuntimeError(self._error_message)
        return "DB_OK"


class TestPidAlive:
    def test_current_process_is_alive(self):
        assert kuzu_worker._pid_alive(os.getpid()) is True

    def test_nonpositive_pid_is_conservative(self):
        assert kuzu_worker._pid_alive(0) is True
        assert kuzu_worker._pid_alive(-1) is True

    def test_process_lookup_error_means_dead(self, monkeypatch):
        def _raise_lookup(pid, sig):
            raise ProcessLookupError()

        monkeypatch.setattr(os, "kill", _raise_lookup)
        assert kuzu_worker._pid_alive(999999) is False

    def test_permission_error_means_alive(self, monkeypatch):
        def _raise_perm(pid, sig):
            raise PermissionError()

        monkeypatch.setattr(os, "kill", _raise_perm)
        assert kuzu_worker._pid_alive(999999) is True


class TestReclaimStaleLock:
    def _lock_file(self, tmp_path):
        db_path = str(tmp_path / "cognee_graph_ladybug")
        lock_path = db_path + ".lock"
        with open(lock_path, "w") as handle:
            handle.write("stale")
        return db_path, lock_path

    def test_removes_lock_when_holder_pid_is_dead(self, tmp_path, monkeypatch):
        db_path, lock_path = self._lock_file(tmp_path)
        monkeypatch.setattr(kuzu_worker, "_pid_alive", lambda pid: False)
        exc = RuntimeError(_LOCK_MSG_TMPL.format(path=db_path, pid=4242))

        assert kuzu_worker._reclaim_stale_lock(db_path, exc) is True
        assert not os.path.exists(lock_path)

    def test_keeps_lock_when_holder_pid_is_alive(self, tmp_path, monkeypatch):
        db_path, lock_path = self._lock_file(tmp_path)
        monkeypatch.setattr(kuzu_worker, "_pid_alive", lambda pid: True)
        exc = RuntimeError(_LOCK_MSG_TMPL.format(path=db_path, pid=4242))

        assert kuzu_worker._reclaim_stale_lock(db_path, exc) is False
        assert os.path.exists(lock_path)

    def test_keeps_lock_for_our_own_pid(self, tmp_path):
        db_path, lock_path = self._lock_file(tmp_path)
        exc = RuntimeError(_LOCK_MSG_TMPL.format(path=db_path, pid=os.getpid()))

        assert kuzu_worker._reclaim_stale_lock(db_path, exc) is False
        assert os.path.exists(lock_path)

    def test_no_pid_in_message_does_not_reclaim(self, tmp_path, monkeypatch):
        db_path, lock_path = self._lock_file(tmp_path)
        monkeypatch.setattr(kuzu_worker, "_pid_alive", lambda pid: False)
        exc = RuntimeError("IO exception: Could not set lock on file : " + db_path)

        assert kuzu_worker._reclaim_stale_lock(db_path, exc) is False
        assert os.path.exists(lock_path)

    def test_missing_lock_file_returns_false(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "cognee_graph_ladybug")  # no .lock created
        monkeypatch.setattr(kuzu_worker, "_pid_alive", lambda pid: False)
        exc = RuntimeError(_LOCK_MSG_TMPL.format(path=db_path, pid=4242))

        assert kuzu_worker._reclaim_stale_lock(db_path, exc) is False

    def test_empty_db_path_returns_false(self):
        exc = RuntimeError(_LOCK_MSG_TMPL.format(path="", pid=4242))
        assert kuzu_worker._reclaim_stale_lock("", exc) is False


class TestOpenLockedWithRecovery:
    @pytest.fixture(autouse=True)
    def fast_retries(self, monkeypatch):
        monkeypatch.setattr(harness, "OPEN_LOCK_RETRIES", 1)
        monkeypatch.setattr(harness, "OPEN_LOCK_BACKOFF", 0.001)

    def test_reopens_after_reclaiming_stale_lock(self, monkeypatch):
        """Backoff exhausts on a dead-worker lock, the stale lock is reclaimed,
        and the following reopen succeeds."""
        msg = _LOCK_MSG_TMPL.format(path="/tmp/x", pid=4242)
        # The initial open already failed (passed in as first_exc). With one
        # backoff retry that also fails, the lock is reclaimed and the reopen
        # succeeds — so the fake sees the retry call + the reopen call.
        fake = _FakeLadybug(fail_times=1, error_message=msg)
        monkeypatch.setattr(kuzu_worker, "_reclaim_stale_lock", lambda db_path, exc: True)

        result = kuzu_worker._open_locked_with_recovery(
            fake, {"database_path": "/tmp/x"}, "/tmp/x", RuntimeError(msg)
        )

        assert result == "DB_OK"
        assert fake.calls == 2  # backoff retry (fail) + reopen after reclaim (ok)

    def test_raises_when_lock_not_reclaimable(self, monkeypatch):
        """A live/unknown holder is not reclaimed, so the lock error propagates."""
        msg = _LOCK_MSG_TMPL.format(path="/tmp/x", pid=4242)
        fake = _FakeLadybug(fail_times=99, error_message=msg)
        monkeypatch.setattr(kuzu_worker, "_reclaim_stale_lock", lambda db_path, exc: False)

        with pytest.raises(RuntimeError, match="Could not set lock on file"):
            kuzu_worker._open_locked_with_recovery(
                fake, {"database_path": "/tmp/x"}, "/tmp/x", RuntimeError(msg)
            )

    def test_non_lock_error_is_not_reclaimed(self, monkeypatch):
        """A non-lock RuntimeError during retry surfaces without touching the lock."""
        fake = _FakeLadybug(fail_times=99, error_message="some other corruption")
        reclaim_calls = []
        monkeypatch.setattr(
            kuzu_worker,
            "_reclaim_stale_lock",
            lambda db_path, exc: reclaim_calls.append(1) or False,
        )

        with pytest.raises(RuntimeError, match="some other corruption"):
            kuzu_worker._open_locked_with_recovery(
                fake,
                {"database_path": "/tmp/x"},
                "/tmp/x",
                RuntimeError(_LOCK_MSG_TMPL.format(path="/tmp/x", pid=4242)),
            )
        assert reclaim_calls == [], "non-lock errors must not trigger reclamation"
