"""A local server still starting at session start is waited for in the background.

A first start on an existing store can migrate for many minutes before the
server answers. Session start blocks only for a short foreground wait; past it,
the session starts, tools say the server is still starting, and the provider
connects once /health answers — without ever starting a second server.
"""

import json
import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import FakeBackend  # noqa: E402
from cognee_integration_hermes import provider as provider_mod  # noqa: E402
from cognee_integration_hermes import server_bootstrap as sb  # noqa: E402


def _alive_proc():
    proc = mock.MagicMock()
    proc.poll.return_value = None
    return proc


def _starting(proc=None):
    return sb.ServerStillStartingError(
        "still starting",
        url="http://127.0.0.1:8011",
        port=8011,
        proc=proc or _alive_proc(),
        log_path="/tmp/server.log",
        remaining=500.0,
    )


class TestForegroundWait:
    def test_a_live_server_past_the_wait_is_still_starting(self):
        proc = _alive_proc()
        clock = iter([0.0, 0.0, 100.0])
        with (
            mock.patch.object(sb, "health_ok", return_value=False),
            mock.patch.object(sb, "server_python", return_value=sys.executable),
            mock.patch.object(sb, "_spawn", return_value=proc) as spawn,
            mock.patch.object(sb, "port_bound", return_value=False),
            mock.patch.object(sb.time, "monotonic", side_effect=lambda: next(clock)),
            mock.patch.object(sb.time, "sleep"),
        ):
            with pytest.raises(sb.ServerStillStartingError) as raised:
                sb.ensure_local_server(8011, log_path="/tmp/server.log", boot_timeout=600, wait=90)
        assert spawn.call_count == 1
        assert raised.value.remaining == pytest.approx(510.0)
        assert raised.value.proc is proc

    def test_a_dead_server_past_the_wait_is_a_plain_failure(self):
        proc = _alive_proc()
        proc.poll.side_effect = [None, 3]
        proc.returncode = 3
        clock = iter([0.0, 0.0, 100.0])
        with (
            mock.patch.object(sb, "health_ok", return_value=False),
            mock.patch.object(sb, "server_python", return_value=sys.executable),
            mock.patch.object(sb, "_spawn", return_value=proc),
            mock.patch.object(sb, "port_bound", return_value=True),
            mock.patch.object(sb.time, "monotonic", side_effect=lambda: next(clock)),
            mock.patch.object(sb.time, "sleep"),
        ):
            with pytest.raises(RuntimeError) as raised:
                sb.ensure_local_server(8011, log_path="/tmp/server.log", boot_timeout=600, wait=90)
        assert not isinstance(raised.value, sb.ServerStillStartingError)

    def test_wait_for_server_reports_ready(self):
        with (
            mock.patch.object(sb, "health_ok", side_effect=[False, True]),
            mock.patch.object(sb.time, "sleep"),
        ):
            assert sb.wait_for_server(_starting()) is True

    def test_wait_for_server_gives_up_when_the_server_died(self):
        proc = _alive_proc()
        proc.poll.return_value = 1
        with (
            mock.patch.object(sb, "health_ok", return_value=False),
            mock.patch.object(sb, "port_bound", return_value=False),
            mock.patch.object(sb, "_spawn") as spawn,
        ):
            assert sb.wait_for_server(_starting(proc)) is False
        spawn.assert_not_called()


def _start(monkeypatch, *, ready):
    for key in ("COGNEE_BASE_URL", "COGNEE_SERVICE_URL", "COGNEE_EMBEDDED"):
        monkeypatch.setenv(key, "")
    gate = threading.Event()

    def wait(starting):
        gate.wait(5)
        return ready

    backend = FakeBackend()
    p = provider_mod.CogneeMemoryProvider(backend=backend)
    with (
        mock.patch.object(provider_mod, "ensure_local_server", side_effect=_starting()),
        mock.patch.object(provider_mod, "wait_for_server", side_effect=wait),
    ):
        p.initialize("sid")
        envelope = json.loads(p._unavailable_envelope())["error"]
        usable_while_starting = p._is_usable()
        gate.set()
        for thread in threading.enumerate():
            if thread.name == "cognee-hermes-server-start":
                thread.join(5)
    return p, backend, envelope, usable_while_starting


def test_session_starts_and_connects_once_the_server_is_ready(monkeypatch):
    p, backend, envelope, usable_while_starting = _start(monkeypatch, ready=True)
    assert "still starting" in envelope
    assert usable_while_starting is False
    assert p._is_usable() is True
    assert backend.only_call("connect")["url"] == "http://127.0.0.1:8011"


def test_a_server_that_never_comes_up_says_why(monkeypatch):
    p, backend, _envelope, _usable = _start(monkeypatch, ready=False)
    assert p._is_usable() is False
    message = json.loads(p._unavailable_envelope())["error"]
    assert "did not become ready" in message and "/tmp/server.log" in message
    assert not backend.kwargs_for("connect")


def test_a_failed_start_names_the_cause(monkeypatch):
    for key in ("COGNEE_BASE_URL", "COGNEE_SERVICE_URL", "COGNEE_EMBEDDED"):
        monkeypatch.setenv(key, "")
    p = provider_mod.CogneeMemoryProvider(backend=FakeBackend())
    cause = RuntimeError("cognee local server exited with code 3. See /tmp/server.log")
    with mock.patch.object(provider_mod, "ensure_local_server", side_effect=cause):
        with pytest.raises(RuntimeError, match="exited with code 3"):
            p.initialize("sid")
