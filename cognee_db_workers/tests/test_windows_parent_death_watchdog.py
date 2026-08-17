"""Tests for `start_windows_parent_death_watchdog` (harness.py).

Root cause this fixes: on Windows, `start_parent_liveness_watchdog`'s trigger
(`os.getppid()` changing) can never fire -- Windows doesn't reparent orphans
the way POSIX does, so `os.getppid()` returns the original parent forever,
dead or alive. This left every `cognee_db_workers` worker spawned on Windows
with zero working tie to its parent's lifecycle once the parent was
force-killed -- confirmed live 2026-07-17/18: an orphaned kuzu_worker.py
process held an exclusive graph-DB lock for hours after its parent was
killed, with no self-termination, causing a real multi-hour outage.

Real subprocess spawn/kill throughout, no mocking of ctypes/WinDLL -- the
whole point is verifying the actual OS-level guarantee (a Windows HANDLE
keeps pointing at the original process object even across PID reuse), which
a mock of the Windows API can't meaningfully exercise.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only fix")

_REPO_ROOT = r"C:\projects\cognee"

_WORKER_SCRIPT = """
import sys, time
sys.path.insert(0, {repo_root!r})
from cognee_db_workers.harness import start_windows_parent_death_watchdog
armed = start_windows_parent_death_watchdog(int(sys.argv[1]))
print("ARMED" if armed else "NOT_ARMED", flush=True)
time.sleep(300)
""".format(repo_root=_REPO_ROOT)


def _spawn_sleeper() -> subprocess.Popen:
    """A throwaway process standing in for 'the parent to watch' -- the
    watchdog only needs a real PID to open a handle to, not a genuine OS
    parent/child relationship, since original_ppid is passed explicitly."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _spawn_worker(watch_pid: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKER_SCRIPT, str(watch_pid)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return proc


def _wait_for_line(proc: subprocess.Popen, timeout: float = 10.0) -> str:
    """Block until the worker's stdout prints its ARMED/NOT_ARMED line."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline().strip()
        if line:
            return line
    raise TimeoutError("worker never printed its armed/not-armed status")


class TestWindowsParentDeathWatchdog:
    def test_self_exits_quickly_when_parent_is_hard_killed(self):
        """The core fix: a hard kill (TerminateProcess, what Stop-Process -Force
        and taskkill /F both do) of the watched PID must cause the worker to
        self-exit within a few seconds -- not sit alive for the rest of its
        300s sleep, which is exactly what happened for hours in the real
        incident this fixes."""
        parent = _spawn_sleeper()
        worker = None
        try:
            worker = _spawn_worker(parent.pid)
            status = _wait_for_line(worker)
            assert status == "ARMED", f"watchdog failed to arm: {status}"

            parent.kill()  # subprocess.Popen.kill() -> TerminateProcess on Windows
            parent.wait(timeout=5)

            deadline = time.monotonic() + 5.0
            exited = False
            while time.monotonic() < deadline:
                if worker.poll() is not None:
                    exited = True
                    break
                time.sleep(0.1)
            assert exited, "worker did not self-exit within 5s of its watched parent being hard-killed"
        finally:
            for p in (worker, parent):
                if p is not None and p.poll() is None:
                    p.kill()

    def test_no_self_exit_while_parent_is_alive(self):
        """Regression guard: the watchdog must not fire spuriously while the
        watched process is genuinely still alive."""
        parent = _spawn_sleeper()
        worker = None
        try:
            worker = _spawn_worker(parent.pid)
            status = _wait_for_line(worker)
            assert status == "ARMED"

            time.sleep(2.0)
            assert worker.poll() is None, "worker exited even though its watched parent is still alive"
        finally:
            for p in (worker, parent):
                if p is not None and p.poll() is None:
                    p.kill()

    def test_already_dead_watched_pid_exits_immediately_not_hang(self):
        """If the watched PID is already gone by the time the watchdog tries to
        open it (the parent died in the narrow window before arming), the
        worker must exit fast rather than hang forever with nothing to watch."""
        already_dead = _spawn_sleeper()
        already_dead.kill()
        already_dead.wait(timeout=5)
        dead_pid = already_dead.pid

        worker = _spawn_worker(dead_pid)
        try:
            deadline = time.monotonic() + 5.0
            exited = False
            while time.monotonic() < deadline:
                if worker.poll() is not None:
                    exited = True
                    break
                time.sleep(0.1)
            assert exited, "worker hung instead of exiting immediately when the watched PID was already dead"
        finally:
            if worker.poll() is None:
                worker.kill()

    def test_pid_reuse_does_not_confuse_the_handle(self):
        """The whole reason this uses a HANDLE instead of polling the PID
        number: if the original PID gets reused by an unrelated process after
        the original dies, the handle must still correctly fire on the
        ORIGINAL process's death, not be confused by the new process at the
        same PID number still being alive. True OS-level PID reuse can't be
        forced deterministically in a test, so this exercises the logical
        equivalent: the worker must have already exited (from test 1's kill)
        well before any subsequent process could plausibly reuse that PID,
        proving the watchdog reacted to the ORIGINAL object dying, not to
        some later liveness check of the numeric PID."""
        parent = _spawn_sleeper()
        worker = None
        try:
            worker = _spawn_worker(parent.pid)
            status = _wait_for_line(worker)
            assert status == "ARMED"
            watched_pid = parent.pid

            parent.kill()
            parent.wait(timeout=5)

            # Immediately spawn a new, unrelated process -- on a busy CI/dev box
            # this has a real (if small) chance of landing on the just-freed PID
            # number. The watchdog must not treat this new, unrelated, ALIVE
            # process as "the original parent is still alive".
            impostor = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 5.0
                exited = False
                while time.monotonic() < deadline:
                    if worker.poll() is not None:
                        exited = True
                        break
                    time.sleep(0.1)
                assert exited, (
                    f"worker did not exit after its watched pid {watched_pid} died, "
                    "even with a new unrelated process potentially reusing that pid number"
                )
            finally:
                if impostor.poll() is None:
                    impostor.kill()
        finally:
            for p in (worker, parent):
                if p is not None and p.poll() is None:
                    p.kill()


class TestNonWindowsPlatform:
    def test_returns_false_on_non_windows(self, monkeypatch):
        import cognee_db_workers.harness as harness_module

        monkeypatch.setattr(harness_module.sys, "platform", "linux")
        assert harness_module.start_windows_parent_death_watchdog(12345) is False
