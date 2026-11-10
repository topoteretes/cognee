"""Live round trip against a real cognee server — the only proof that matters.

Everything else in this suite is mocked at one seam or another, so nothing else
can catch a wire format the server rejects, or a field it accepts and ignores.
This is specifically the test for the bug the HTTP transport exists to fix:
``improve(session_ids=[...])`` bridging a session's cached turns into the
permanent graph. Through the SDK's ``CloudClient`` those ids are dropped
(``improve.py:128`` forwards only ``node_name`` and ``**kwargs``), so the bridge
silently degraded into a dataset-wide improve and session turns never landed in
the graph.

**Opt-in**: needs a real cognee install *and* LLM credentials (graph extraction
calls an LLM), so it is skipped unless both are present::

    COGNEE_RUN_INTEGRATION=1 LLM_API_KEY=sk-... \\
        uv run pytest tests/test_integration_roundtrip.py -v

Each run uses its own dataset and its own port, so repeated runs do not pollute
each other or a developer's real memory.
"""

import importlib.util
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RUN = os.environ.get("COGNEE_RUN_INTEGRATION") == "1"
_HAS_COGNEE = importlib.util.find_spec("cognee") is not None
_HAS_LLM = bool(os.environ.get("LLM_API_KEY"))
_REASON = (
    "set COGNEE_RUN_INTEGRATION=1, install cognee, and provide LLM_API_KEY "
    "(graph extraction needs an LLM) to run the live round trip"
)

# Graph building is LLM-bound; these are deliberately patient.
_BOOT_TIMEOUT = 120.0
_WRITE_TIMEOUT = 600.0
_IMPROVE_TIMEOUT = 900.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _unique_suffix() -> str:
    # Unique per run so reruns neither collide nor accumulate.
    return f"{os.getpid()}_{int(time.time())}"


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


@unittest.skipUnless(_RUN and _HAS_COGNEE and _HAS_LLM, _REASON)
class TestHttpRoundTrip(unittest.TestCase):
    """remember -> recall -> session turn -> session end -> recall from the graph."""

    @classmethod
    def setUpClass(cls):
        from cognee_integration_hermes.server_bootstrap import ensure_local_server

        cls.port = _free_port()
        cls.home = tempfile.TemporaryDirectory()
        # Own the server's storage too, so the run leaves nothing behind.
        cls.url = ensure_local_server(
            cls.port,
            data_root=str(Path(cls.home.name) / "cognee" / "data"),
            system_root=str(Path(cls.home.name) / "cognee" / "system"),
            boot_timeout=_BOOT_TIMEOUT,
        )

    @classmethod
    def tearDownClass(cls):
        _kill_server_on(cls.port)
        cls.home.cleanup()

    def setUp(self):
        # One dataset and one session per test. Sharing them let the forget test
        # (alphabetically earlier) delete the state the bridge test needs, which
        # surfaced as a misleading "Recall prerequisites not met" from the server.
        self.dataset = f"hermes_it_{_unique_suffix()}_{self._testMethodName[5:25]}"

    def _provider(self):
        from cognee_integration_hermes.http_backend import HttpBackend
        from cognee_integration_hermes.provider import CogneeMemoryProvider

        env = {
            "COGNEE_BASE_URL": "",
            "COGNEE_SERVICE_URL": "",
            "COGNEE_EMBEDDED": "",
            "COGNEE_LOCAL_PORT": str(self.port),
            "COGNEE_DATASET": self.dataset,
            # Synchronous improve: on_session_end backgrounds it by default in
            # server mode, and a backgrounded bridge cannot be asserted on.
            "COGNEE_IMPROVE_BACKGROUND": "false",
            "COGNEE_WRITE_TIMEOUT": str(int(_WRITE_TIMEOUT)),
            "COGNEE_IMPROVE_TIMEOUT": str(int(_IMPROVE_TIMEOUT)),
            "COGNEE_RECALL_TIMEOUT": "120",
        }
        provider = CogneeMemoryProvider()
        # No key-cache patch: the HTTP backend keeps a resolved key in-process
        # (http_backend._RESOLVED_KEYS) and writes no api_key.json any more, so
        # a test server on a random port cannot touch the machine's real key.
        with mock.patch.dict("os.environ", env, clear=False):
            provider.initialize(f"rt-{self._testMethodName}", hermes_home=self.home.name)
        self.assertIsInstance(
            provider._backend, HttpBackend, "the HTTP transport should be the default"
        )
        self.assertTrue(provider._initialized)
        return provider

    def test_explicit_remember_is_recallable(self):
        import json

        provider = self._provider()
        try:
            fact = "The Meditech Solutions contract is worth 1.2 million pounds."
            stored = json.loads(provider.handle_tool_call("cognee_remember", {"content": fact}))
            self.assertNotIn("error", stored, stored)

            found = json.loads(
                provider.handle_tool_call(
                    "cognee_recall",
                    {"query": "What is the Meditech Solutions contract worth?"},
                )
            )
            self.assertNotIn("error", found, found)
            self.assertGreater(found.get("count", 0), 0, f"nothing recalled: {found}")
        finally:
            provider.shutdown()

    def test_session_turns_reach_the_graph_after_session_end(self):
        """The regression test for the dropped ``session_ids``.

        A turn goes into the session cache, which does no graph extraction. Only
        ``improve(session_ids=[...])`` promotes it. If those ids do not reach the
        server the bridge becomes a dataset-wide improve and the turn stays
        invisible to a graph-scoped recall.
        """
        import json

        provider = self._provider()
        try:
            provider.sync_turn(
                "Remember that Ada Lovelace wrote the first algorithm.",
                "Noted — Ada Lovelace wrote the first algorithm.",
            )
            if provider._sync_thread is not None:
                provider._sync_thread.join(timeout=_WRITE_TIMEOUT)

            # on_session_end logs improve failures and carries on, so watch the
            # log rather than trusting silence.
            with self.assertLogs("cognee_integration_hermes.provider", level="WARNING") as logs:
                provider.on_session_end([])
                logging.getLogger("cognee_integration_hermes.provider").warning(
                    "sentinel: assertLogs requires at least one record"
                )
            improve_failures = [line for line in logs.output if "improve failed" in line]
            self.assertEqual(improve_failures, [], f"improve() failed: {improve_failures}")

            found = json.loads(
                provider.handle_tool_call(
                    "cognee_recall",
                    {"query": "Who wrote the first algorithm?"},
                )
            )
            self.assertNotIn("error", found, found)
            self.assertGreater(
                found.get("count", 0),
                0,
                "the session turn never reached the graph — improve() probably did "
                f"not carry session_ids: {found}",
            )
        finally:
            provider.shutdown()

    def test_forget_finds_and_deletes_the_stored_document(self):
        import json

        provider = self._provider()
        try:
            provider.handle_tool_call(
                "cognee_remember", {"content": "Disposable fact about zebras."}
            )
            # cognee_remember runs in the background (#479): the document joins
            # the dataset a moment after the call returns, so wait for it.
            deadline = time.monotonic() + 120
            while True:
                found = json.loads(
                    provider.handle_tool_call(
                        "cognee_forget", {"action": "find", "terms": "zebras"}
                    )
                )
                if found.get("candidates") or time.monotonic() > deadline:
                    break
                time.sleep(2)
            self.assertNotIn("error", found, found)
            candidates = found.get("candidates") or []
            self.assertTrue(candidates, f"the stored document was not listed: {found}")
            data_ids = [candidate["data_id"] for candidate in candidates]
            result = json.loads(
                provider.handle_tool_call(
                    "cognee_forget",
                    {"action": "forget", "data_ids": data_ids, "confirm": True},
                )
            )
            self.assertNotIn("error", result, result)
            self.assertEqual(result["count"], len(data_ids), result)
        finally:
            provider.shutdown()


@unittest.skipUnless(
    _RUN and _HAS_COGNEE,
    "set COGNEE_RUN_INTEGRATION=1 and install cognee to run the live connect check",
)
class TestHttpConnectAgainstRealServer(unittest.TestCase):
    """Boot a real server and exercise ``connect()`` — no LLM required.

    Covers the parts that only a real server can confirm: that ``/health`` answers,
    that an API key can actually be minted through
    ``/api/v1/auth/login`` -> ``/api/v1/auth/api-keys``, and that the agent
    connection registers and unregisters. Writes and recalls need an LLM, so they
    live in :class:`TestHttpRoundTrip`; this class is the credential-free half and
    is the one to run first when something looks wrong.
    """

    def test_connect_finds_the_servers_key_and_registers(self):
        from cognee_integration_hermes.http_backend import HttpBackend
        from cognee_integration_hermes.server_bootstrap import ensure_local_server

        port = _free_port()
        self.addCleanup(_kill_server_on, port)
        with tempfile.TemporaryDirectory() as home:
            url = ensure_local_server(
                port,
                data_root=str(Path(home) / "data"),
                system_root=str(Path(home) / "system"),
                boot_timeout=_BOOT_TIMEOUT,
            )
            backend = HttpBackend()
            try:
                backend.connect(url=url, api_key="", timeout=60.0)
                self.assertEqual(backend.url, url)
                self.assertTrue(backend.registered, "agent connection should register")
                # A server with auth enabled hands back a key; one with auth
                # disabled needs none. A second connection must land on the same
                # key, looked up on the server rather than in a file.
                if backend.api_key:
                    again = HttpBackend()
                    again.connect(url=url, api_key="", timeout=60.0)
                    try:
                        self.assertEqual(again.api_key, backend.api_key)
                    finally:
                        again.close(timeout=15.0)
            finally:
                backend.close(timeout=15.0)
            self.assertFalse(backend.registered, "close() should unregister")


@unittest.skipUnless(_RUN, "set COGNEE_RUN_INTEGRATION=1 to run against real sockets")
class TestHttpConnectFailsLoudly(unittest.TestCase):
    """An unreachable target must raise, not look like success.

    ``cognee.serve()`` logged a warning and returned a client anyway, so a wrong
    ``COGNEE_BASE_URL`` only surfaced on the first real call. This needs neither
    cognee nor an LLM — just a real socket refusing a connection — so it is the
    cheapest live check in the suite. Its mocked twin lives in
    ``test_http_backend.py::TestConnect``.
    """

    def test_unreachable_base_url_raises(self):
        from cognee_integration_hermes.provider import CogneeMemoryProvider

        env = {
            # A port nothing is listening on.
            "COGNEE_BASE_URL": f"http://127.0.0.1:{_free_port()}",
            "COGNEE_SERVICE_URL": "",
            "COGNEE_EMBEDDED": "",
        }
        provider = CogneeMemoryProvider()
        with mock.patch.dict("os.environ", env, clear=False):
            with self.assertRaises(RuntimeError):
                provider.initialize("unreachable-1")
        self.assertFalse(provider._initialized)

        # And having failed, it must refuse to work rather than degrade.
        import json

        out = json.loads(provider.handle_tool_call("cognee_recall", {"query": "q"}))
        self.assertIn("unavailable", out["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
