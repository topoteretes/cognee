"""Parent-death checks must leave workers running when liveness is uncertain."""

import multiprocessing
import os
import sys
from multiprocessing.process import _ParentProcess

import pytest

from cognee_db_workers import harness

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor semantics")


class TestParentAlreadyExitedUnitSemantics:
    """Fail-safe contract of the helpers, without spawning anything."""

    def test_returns_false_without_a_baseline(self):
        from cognee_db_workers import harness

        # Not a multiprocessing child at all: no sentinel, no baseline pid.
        # Must NOT claim the parent is dead.
        assert harness.get_original_parent_pid() is None
        assert harness.parent_already_exited(None) is False

    def test_returns_false_on_win32_regardless_of_baseline(self, monkeypatch):
        from cognee_db_workers import harness

        monkeypatch.setattr(harness.sys, "platform", "win32")
        assert harness.parent_already_exited(999999) is False
        assert harness.parent_already_exited(None) is False

    def test_sentinel_absent_reports_unknown_not_dead(self):
        from cognee_db_workers import harness

        # In the main process there is no parent sentinel; the tri-state must
        # be None ("cannot tell"), never False ("dead").
        assert harness._parent_sentinel_alive() is None

    def test_there_is_no_getppid_fallback_at_all(self, monkeypatch):
        """No usable sentinel must mean "unknown", never "dead".

        REPLACES test_getppid_fallback_is_not_used_under_forkserver, which
        asserted the fallback was *gated* correctly. The fallback has since been
        REMOVED outright: its gate read `get_start_method()`, i.e. the
        process-wide DEFAULT context rather than the context that created this
        child, so a process defaulting to spawn while launching via forkserver
        would pass the gate and then evaluate a comparison that differs BY
        DESIGN -- killing every healthy worker at startup.

        A gate on the wrong fact is not a gate. This asserts the property that
        replaced it: with no answer from the sentinel, the result is False
        (do not kill) regardless of pids or start method.
        """
        import multiprocessing

        from cognee_db_workers import harness

        monkeypatch.setattr(harness, "_parent_sentinel_alive", lambda: None)
        bogus_baseline = 999999
        assert os.getppid() != bogus_baseline, "baseline must differ for this to mean anything"

        # Under EVERY start method -- including the ones the old fallback
        # treated as safe -- an unanswerable sentinel must never yield "dead".
        for method in ("spawn", "fork", "forkserver"):
            monkeypatch.setattr(
                multiprocessing, "get_start_method", lambda allow_none=False, _m=method: _m
            )
            assert harness.parent_already_exited(bogus_baseline) is False, (
                f"under {method}: no sentinel answer must mean unknown, not dead. "
                "Returning True here kills healthy workers."
            )


@pytest.fixture
def parent_sentinel(monkeypatch):
    reader, writer = os.pipe()
    parent = _ParentProcess("test parent", os.getpid(), reader)
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: parent)
    try:
        yield parent, writer
    finally:
        for fd in (reader, writer):
            try:
                os.close(fd)
            except OSError:
                pass


def test_live_parent_sentinel(parent_sentinel):
    assert harness._parent_sentinel_alive() is True
    assert harness.parent_already_exited(os.getpid()) is False


def test_closed_sentinel_is_unknown(parent_sentinel):
    parent, _ = parent_sentinel
    os.close(parent.sentinel)
    assert harness._parent_sentinel_alive() is None
    assert harness.parent_already_exited(os.getpid()) is False


@pytest.mark.parametrize("replacement_kind", ["file", "pipe"])
def test_reused_sentinel_does_not_report_healthy_parent_dead(
    parent_sentinel, tmp_path, replacement_kind
):
    parent, _ = parent_sentinel
    if replacement_kind == "file":
        replacement = os.open(tmp_path / "replacement", os.O_CREAT | os.O_RDONLY, 0o600)
    else:
        replacement, writer = os.pipe()
        os.close(writer)
    try:
        os.dup2(replacement, parent.sentinel)
    finally:
        os.close(replacement)

    # Reproduce both premises of the old guard using real kernel descriptors.
    # Neither a poll result nor a successful fstat identifies the original pipe.
    assert parent.is_alive() is False
    os.fstat(parent.sentinel)
    assert harness._parent_sentinel_alive() is None
    assert harness.parent_already_exited(parent.pid) is False


def test_sentinel_death_requires_independent_confirmation(parent_sentinel, monkeypatch):
    parent, writer = parent_sentinel
    os.close(writer)

    def missing_process(pid, sig):
        assert (pid, sig) == (parent.pid, 0)
        raise ProcessLookupError

    monkeypatch.setattr(harness.os, "kill", missing_process)
    assert harness._parent_sentinel_alive() is False
    assert harness.parent_already_exited(parent.pid) is True


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_unavailable_pid_probe_leaves_sentinel_death_unknown(parent_sentinel, monkeypatch, error):
    parent, writer = parent_sentinel
    os.close(writer)

    def unavailable_process(pid, sig):
        raise error

    monkeypatch.setattr(harness.os, "kill", unavailable_process)
    assert harness._parent_sentinel_alive() is None
    assert harness.parent_already_exited(parent.pid) is False


@pytest.mark.parametrize("pid", [None, 0, -1, "123"])
def test_invalid_pid_cannot_confirm_death(monkeypatch, pid):
    def unexpected_probe(*args):
        pytest.fail("invalid PID must never reach os.kill")

    monkeypatch.setattr(harness.os, "kill", unexpected_probe)
    assert harness._parent_pid_exited(pid) is False


@pytest.mark.parametrize(
    "proc_stat,exited",
    [
        (b"123 (worker (with spaces)) Z 1 2 3", True),
        (b"123 (worker) X 1 2 3", True),
        (b"123 (worker) S 1 2 3", False),
        (b"malformed stat", False),
        (b"123 (worker)", False),
    ],
)
def test_linux_process_state(monkeypatch, proc_stat, exited):
    from io import BytesIO

    monkeypatch.setattr(harness.sys, "platform", "linux")
    monkeypatch.setattr(harness.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: BytesIO(proc_stat))
    assert harness._parent_pid_exited(123) is exited


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError, OSError])
def test_unavailable_procfs_does_not_confirm_death(monkeypatch, error):
    monkeypatch.setattr(harness.sys, "platform", "linux")
    monkeypatch.setattr(harness.os, "kill", lambda pid, sig: None)

    def unavailable_stat(*args, **kwargs):
        raise error

    monkeypatch.setattr("builtins.open", unavailable_stat)
    assert harness._parent_pid_exited(123) is False
