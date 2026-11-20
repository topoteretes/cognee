"""Integration tests for the core motivation of local-server mode: a real server
spawn, and concurrent operations that must NOT raise "database is locked".

These require the full cognee stack (and LLM creds for the write path), so they
are **opt-in**: set ``COGNEE_RUN_INTEGRATION=1`` to run them. They are skipped in
the default unit run (and in CI, which has neither cognee installed nor creds).
The unit suite in ``test_server_mode.py`` covers the routing logic with mocks;
this file covers the behavior that can only be observed against a live server.

Run locally (needs cognee installed + LLM creds for the write path):
    COGNEE_RUN_INTEGRATION=1 uv run pytest tests/test_integration_concurrency.py
"""

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RUN = os.environ.get("COGNEE_RUN_INTEGRATION") == "1"
_HAS_COGNEE = importlib.util.find_spec("cognee") is not None
_REASON = "set COGNEE_RUN_INTEGRATION=1 and install cognee to run integration tests"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill_server_on(port: int) -> None:
    """Stop the test-spawned server. ensure_local_server detaches it on purpose
    (it must outlive Hermes in production), so tests have to reap it by port or
    it lingers after the run, contending for whatever store it was pointed at."""
    try:
        found = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for pid in found.stdout.split():
            os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass


def _isolated_env(tmp: str, port: int) -> dict:
    """Point the spawned server and the provider at test-owned storage.

    Without this the server would land on the shared ~/.cognee — the store a
    developer's real cognee plugins are using — and two servers on one
    single-writer store is exactly the contention these tests exist to rule out.
    """
    return {
        "COGNEE_LOCAL_PORT": str(port),
        "COGNEE_DATA_ROOT": str(Path(tmp) / "data"),
        "COGNEE_SYSTEM_ROOT": str(Path(tmp) / "system"),
        "COGNEE_BASE_URL": "",
        "COGNEE_SERVICE_URL": "",
        "COGNEE_EMBEDDED": "",
    }


def _looks_like_lock_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "database is locked" in text or "database table is locked" in text


@unittest.skipUnless(_RUN and _HAS_COGNEE, _REASON)
class TestRealServerSpawn(unittest.TestCase):
    def test_ensure_local_server_spawns_and_serves_health(self):
        from cognee_integration_hermes import server_bootstrap as sb

        port = _free_port()
        tmp = tempfile.TemporaryDirectory()
        # LIFO: kill the server first, then remove the store it was using.
        self.addCleanup(tmp.cleanup)
        self.addCleanup(_kill_server_on, port)
        url = sb.ensure_local_server(
            port,
            data_root=str(Path(tmp.name) / "data"),
            system_root=str(Path(tmp.name) / "system"),
            boot_timeout=90.0,
        )
        self.assertEqual(url, f"http://127.0.0.1:{port}")
        self.assertTrue(sb.health_ok(url), "server should answer /health after spawn")


@unittest.skipUnless(_RUN and _HAS_COGNEE, _REASON)
class TestConcurrentNoLocking(unittest.TestCase):
    def test_concurrent_remember_recall_no_database_locked(self):
        from cognee_integration_hermes import CogneeMemoryProvider

        port = _free_port()
        tmp = tempfile.TemporaryDirectory()
        # LIFO: shutdown (unregister) -> kill the server -> remove its store.
        self.addCleanup(tmp.cleanup)
        self.addCleanup(_kill_server_on, port)
        provider = CogneeMemoryProvider()
        with mock.patch.dict("os.environ", _isolated_env(tmp.name, port), clear=False):
            provider.initialize("integration-concurrency")
        self.addCleanup(provider.shutdown)

        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                provider.handle_tool_call(
                    "cognee_remember",
                    {"content": f"concurrency probe {i}: the sky is blue"},
                )
                provider.handle_tool_call("cognee_recall", {"query": "what colour is the sky"})
            except BaseException as exc:  # noqa: BLE001 — we inspect every failure
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(16)))

        lock_errors = [e for e in errors if _looks_like_lock_error(e)]
        self.assertEqual(lock_errors, [], f"got DB-lock errors under concurrency: {lock_errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
