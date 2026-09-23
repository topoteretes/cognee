"""Windows: a DB worker must exit when its parent is hard-killed."""

import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only watchdog")

WORKER = (
    "import time\n"
    "from cognee_db_workers.harness import start_windows_parent_watchdog\n"
    "assert start_windows_parent_watchdog()\n"
    "time.sleep(60)\n"
)
PARENT = (
    "import multiprocessing as mp, sys, time\n"
    "if __name__ == '__main__':\n"
    "    p = mp.get_context('spawn').Process(target=exec, args=(sys.argv[1], {}))\n"
    "    p.start()\n"
    "    print(p.pid, flush=True)\n"
    "    time.sleep(60)\n"
)


def _spawn():
    import _winapi

    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT, WORKER], stdout=subprocess.PIPE, text=True
    )
    worker = _winapi.OpenProcess(_winapi.SYNCHRONIZE | 0x0001, False, int(parent.stdout.readline()))
    return parent, worker


def _cleanup(parent, worker):
    import _winapi

    if parent.poll() is None:
        parent.kill()
    if _winapi.WaitForSingleObject(worker, 0) != 0:
        _winapi.TerminateProcess(worker, 1)
    _winapi.CloseHandle(worker)


def test_worker_exits_when_parent_is_hard_killed():
    import _winapi

    parent, worker = _spawn()
    try:
        time.sleep(2)
        assert _winapi.WaitForSingleObject(worker, 0) != 0, "worker died while its parent was alive"
        parent.kill()
        assert _winapi.WaitForSingleObject(worker, 10_000) == 0, "worker outlived its parent"
    finally:
        _cleanup(parent, worker)


def test_not_a_multiprocessing_child_returns_false():
    from cognee_db_workers.harness import start_windows_parent_watchdog

    assert start_windows_parent_watchdog() is False
