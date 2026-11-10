"""Automatic capture: tool traces, the capture policy, redaction, and the write buffer.

``sync_turn`` stores the turn's tool calls (from the ``messages`` Hermes passes)
as ``trace`` entries and the turn itself as a ``qa`` entry, after the capture
policy and secret redaction (the claude-code / codex / openclaw rules). A write
the server never received is queued and replayed once it answers again.
"""

import json
import socket
import sys
import time
import unittest
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import FakeBackend, fake_backend, make_provider  # noqa: E402
from cognee_integration_hermes import write_buffer  # noqa: E402
from cognee_integration_hermes.http_backend import (  # noqa: E402
    CogneeHttpError,
    CogneeUnreachable,
    _never_connected,
)

_SECRET = "sk-proj-" + "a" * 24


def _turn(*calls, prior=()):
    """An OpenAI-style message list ending with one user turn and its tool calls."""
    messages = [*prior, {"role": "user", "content": "do the thing"}]
    tool_calls = []
    results = []
    for index, (name, args, result) in enumerate(calls):
        call_id = f"call_{index}"
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
        results.append({"role": "tool", "tool_call_id": call_id, "content": result})
    if tool_calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        messages.extend(results)
    messages.append({"role": "assistant", "content": "done"})
    return messages


def _sync(provider, user="hi", answer="hello", messages=None):
    provider.sync_turn(user, answer, messages=messages)
    if provider._sync_thread is not None:
        provider._sync_thread.join(timeout=5.0)
    _settle_replays()


def _settle_replays():
    """Wait for background replays a successful write started (U5: own thread)."""
    import threading

    for thread in threading.enumerate():
        if thread.name == "cognee-hermes-replay":
            thread.join(timeout=5.0)


class TestTraceCapture(unittest.TestCase):
    def test_tool_calls_become_trace_entries_before_the_turn(self):
        with fake_backend() as fake:
            provider = make_provider()
            _sync(
                provider,
                messages=_turn(
                    ("read_file", {"path": "src/app.py"}, '{"content": "print(1)"}'),
                    ("terminal", {"command": "ls"}, '{"error": "boom"}'),
                ),
            )
            traces = fake.kwargs_for("remember_entry")
            names = fake.names()
        self.assertEqual([t["entry"]["origin_function"] for t in traces], ["read_file", "terminal"])
        first, second = (t["entry"] for t in traces)
        self.assertEqual(first["type"], "trace")
        self.assertEqual(first["status"], "success")
        self.assertEqual(first["method_params"], {"path": "src/app.py"})
        self.assertIn("print(1)", first["method_return_value"])
        self.assertIs(first["generate_feedback_with_llm"], False)
        self.assertEqual(second["status"], "error")
        self.assertEqual(second["error_message"], "boom")
        for trace in traces:
            self.assertEqual(trace["session_id"], "hermes_s-1")
            self.assertEqual(trace["dataset"], "hermes")
        # Traces land before the turn's own QA entry.
        self.assertLess(names.index("remember_entry"), names.index("remember_session"))

    def test_only_the_latest_turn_is_traced(self):
        earlier = _turn(("read_file", {"path": "old.py"}, "{}"))
        with fake_backend() as fake:
            provider = make_provider()
            _sync(provider, messages=_turn(("read_file", {"path": "new.py"}, "{}"), prior=earlier))
            paths = [t["entry"]["method_params"]["path"] for t in fake.kwargs_for("remember_entry")]
        self.assertEqual(paths, ["new.py"])

    def test_no_messages_means_no_traces(self):
        with fake_backend() as fake:
            provider = make_provider()
            _sync(provider)
            self.assertEqual(fake.kwargs_for("remember_entry"), [])
            self.assertEqual(len(fake.kwargs_for("remember_session")), 1)

    def test_secret_files_and_self_references_are_never_traced(self):
        with fake_backend() as fake:
            provider = make_provider()
            _sync(
                provider,
                messages=_turn(
                    ("read_file", {"path": "/repo/.env"}, "SECRET=1"),
                    ("terminal", {"command": "cat ~/.aws/credentials"}, "x"),
                    ("terminal", {"command": "hermes cognee status"}, "x"),
                    ("cognee_recall", {"query": "q"}, "{}"),
                    ("read_file", {"path": "ok.py"}, "{}"),
                ),
            )
            traced = [t["entry"]["origin_function"] for t in fake.kwargs_for("remember_entry")]
        self.assertEqual(traced, ["read_file"])

    def test_deny_list_from_config(self):
        with fake_backend() as fake:
            provider = make_provider(config={"capture_deny_tools": "terminal(git:*)|browser_*"})
            _sync(
                provider,
                messages=_turn(
                    ("terminal", {"command": "git status"}, "{}"),
                    ("browser_navigate", {"url": "https://x"}, "{}"),
                    ("Terminal", {"command": "ls"}, "{}"),
                ),
            )
            traced = [t["entry"]["origin_function"] for t in fake.kwargs_for("remember_entry")]
        self.assertEqual(traced, ["Terminal"])

    def test_secrets_are_redacted_before_truncation(self):
        with fake_backend() as fake:
            provider = make_provider()
            _sync(
                provider,
                user=f"my key is {_SECRET}",
                answer="stored token=hunter2",
                messages=_turn(
                    ("terminal", {"command": f"echo {_SECRET}", "api_key": "x"}, "y" * 20000)
                ),
            )
            (trace,) = (t["entry"] for t in fake.kwargs_for("remember_entry"))
            qa = fake.only_call("remember_session")["text"]
        self.assertNotIn(_SECRET, json.dumps(trace))
        self.assertEqual(trace["method_params"]["api_key"], "[redacted:credential]")
        self.assertTrue(trace["method_return_value"].endswith("…[truncated]"))
        self.assertNotIn(_SECRET, qa)
        self.assertNotIn("hunter2", qa)
        self.assertIn("[redacted:", qa)

    def test_an_unparsable_pattern_stores_nothing(self):
        with fake_backend() as fake:
            provider = make_provider(config={"capture_tools": "terminal(rg)"})
            _sync(provider, messages=_turn(("read_file", {"path": "a.py"}, "{}")))
            self.assertEqual(fake.kwargs_for("remember_entry"), [])
            self.assertEqual(fake.kwargs_for("remember_session"), [])

    def test_capture_off_stores_nothing(self):
        with fake_backend() as fake:
            provider = make_provider(config={"capture": "false"})
            _sync(provider, messages=_turn(("read_file", {"path": "a.py"}, "{}")))
            self.assertEqual(fake.calls, [])

    def test_explicit_remember_is_not_redacted(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider.handle_tool_call("cognee_remember", {"content": "token=keep-me"})
            self.assertEqual(fake.only_call("remember_permanent")["text"], "token=keep-me")

    def test_memory_write_mirror_is_redacted(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider.on_memory_write("add", "user", f"key {_SECRET}")
            self.assertTrue(fake.wait("remember_permanent"))
            text = fake.only_call("remember_permanent")["text"]
        self.assertNotIn(_SECRET, text)

    def test_the_sdk_transport_without_entries_still_stores_the_turn(self):
        class NoEntries(FakeBackend):
            def remember_entry(self, **kwargs):
                raise RuntimeError("requires the HTTP transport")

        backend = NoEntries()
        provider = make_provider(backend=backend)
        _sync(provider, messages=_turn(("read_file", {"path": "a.py"}, "{}")))
        self.assertEqual(len(backend.kwargs_for("remember_session")), 1)
        self.assertEqual(provider._consecutive_failures, 0)


class _ServerBackend(FakeBackend):
    """A fake whose server outlives the process, so writes can be buffered."""

    def __init__(self):
        super().__init__()
        self.down = True

    def connection_info(self):
        return {"url": "http://127.0.0.1:8011", "api_key": "", "agent_session_name": "hermes"}

    def _write(self, name, kwargs):
        result = self._recorder.record(name, kwargs)
        if self.down:
            raise CogneeUnreachable("connection refused", not_connected=True)
        return result

    def remember_session(self, **kwargs):
        return self._write("remember_session", kwargs)

    def remember_entry(self, **kwargs):
        return self._write("remember_entry", kwargs)


class TestProviderBuffering(unittest.TestCase):
    def test_writes_the_server_never_received_are_queued_and_replayed(self):
        backend = _ServerBackend()
        provider = make_provider(backend=backend)
        _sync(provider, user="first", messages=_turn(("read_file", {"path": "a.py"}, "{}")))
        buffer = write_buffer.write_buffer_for(provider._write_buffer_path)
        self.assertEqual(buffer.size(), 2)

        backend.down = False
        _sync(provider, user="second")
        self.assertEqual(buffer.size(), 0)
        # Over HTTP a turn is a typed QA entry: the queued "first" turn was
        # replayed, and the "second" one stored directly, once each.
        stored = [
            kw["entry"]["question"]
            for kw in backend.kwargs_for("remember_entry")
            if kw["entry"]["type"] == "qa"
        ]
        # The trace's refusal ended the first turn, so its QA entry was queued
        # without being tried: it reaches the server once, by replay.
        self.assertEqual(stored.count("first"), 1)
        self.assertEqual(stored.count("second"), 1)

    def test_session_end_replays_the_session_before_improve(self):
        backend = _ServerBackend()
        provider = make_provider(backend=backend)
        _sync(provider, user="lost turn")
        backend.down = False
        provider.on_session_end([])
        names = backend.names()
        self.assertIn("improve", names)
        replay_at = max(i for i, n in enumerate(names) if n == "remember_entry")
        self.assertLess(replay_at, names.index("improve"))
        self.assertEqual(write_buffer.write_buffer_for(provider._write_buffer_path).size(), 0)

    def test_an_open_breaker_queues_instead_of_dropping(self):
        backend = _ServerBackend()
        provider = make_provider(backend=backend)
        provider._is_breaker_open = lambda: True
        _sync(provider, user="while paused")
        self.assertEqual(backend.kwargs_for("remember_session"), [])
        self.assertEqual(write_buffer.write_buffer_for(provider._write_buffer_path).size(), 1)

    def test_a_timeout_is_not_queued(self):
        backend = _ServerBackend()
        provider = make_provider(backend=backend)

        def timed_out(**kwargs):
            backend._recorder.record("remember_entry", kwargs)
            raise CogneeUnreachable("timed out", not_connected=False)

        backend.remember_entry = timed_out
        _sync(provider)
        self.assertEqual(write_buffer.write_buffer_for(provider._write_buffer_path).size(), 0)


class TestOneTurnOneDeadline(unittest.TestCase):
    """A turn's writes share one deadline and give the breaker one verdict."""

    _CALLS = [("read_file", {"path": f"f{i}.py"}, "{}") for i in range(5)]

    def _backend(self, behaviour):
        """A server-backed fake whose writes run ``behaviour(kind, kwargs)``."""
        backend = _ServerBackend()
        backend.down = False

        def write(kind):
            def call(**kwargs):
                backend._recorder.record(kind, kwargs)
                return behaviour(kind, kwargs)

            return call

        backend.remember_entry = write("remember_entry")
        backend.remember_session = write("remember_session")
        return backend

    def _buffered(self, provider):
        return write_buffer.write_buffer_for(provider._write_buffer_path).size()

    def test_a_hung_server_costs_one_timeout_and_queues_the_rest(self):
        def hang(kind, kwargs):
            raise CogneeUnreachable("timed out", not_connected=False)

        backend = self._backend(hang)
        provider = make_provider(backend=backend)
        _sync(provider, messages=_turn(*self._CALLS))
        # One write was tried; the other four traces and the QA entry were never
        # sent, so they wait for the server. The one that timed out is not queued.
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(self._buffered(provider), 5)
        self.assertEqual(provider._consecutive_failures, 1)

    def test_every_write_shares_the_turns_deadline(self):
        def slow(kind, kwargs):
            time.sleep(0.15)

        backend = self._backend(slow)
        provider = make_provider(backend=backend, config={"write_timeout": 0.5})
        started = time.monotonic()
        _sync(provider, messages=_turn(*self._CALLS))
        self.assertLess(time.monotonic() - started, 1.0)
        timeouts = [kwargs["timeout"] for _name, kwargs in backend.calls]
        self.assertTrue(all(b < a for a, b in zip(timeouts, timeouts[1:])), timeouts)
        self.assertLessEqual(timeouts[0], 0.5)
        # What the deadline cut off is queued, not dropped, and not replayed in
        # the same breath: a server too slow for the deadline is a failure.
        self.assertEqual(len(backend.calls) + self._buffered(provider), 6)
        self.assertGreater(self._buffered(provider), 0)
        self.assertEqual(provider._consecutive_failures, 1)

    def test_a_breaker_that_opens_mid_turn_stops_the_writes(self):
        backend = self._backend(lambda kind, kwargs: None)
        provider = make_provider(backend=backend)
        # sync_turn checks once before starting the writes; then two writes go.
        checks = iter([False, False, False])
        provider._is_breaker_open = lambda: next(checks, True)
        _sync(provider, messages=_turn(*self._CALLS))
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(self._buffered(provider), 4)

    def test_refused_writes_are_one_failure_and_the_rest_still_go(self):
        def refuse(kind, kwargs):
            raise CogneeHttpError(400, "bad entry")

        backend = self._backend(refuse)
        provider = make_provider(backend=backend)
        _sync(provider, messages=_turn(*self._CALLS))
        self.assertEqual(len(backend.calls), 6)
        self.assertEqual(self._buffered(provider), 0)
        self.assertEqual(provider._consecutive_failures, 1)

    def test_stored_traces_do_not_cover_a_lost_turn(self):
        def qa_fails(kind, kwargs):
            if kwargs["entry"]["type"] == "qa":
                raise CogneeHttpError(500, "boom")

        backend = self._backend(qa_fails)
        provider = make_provider(backend=backend)
        provider._consecutive_failures = 2
        _sync(provider, messages=_turn(*self._CALLS[:2]))
        # The traces landed, but the turn did not: a failure, not a success.
        self.assertEqual(provider._consecutive_failures, 3)

    def test_a_clean_turn_is_one_success(self):
        backend = self._backend(lambda kind, kwargs: None)
        provider = make_provider(backend=backend)
        provider._consecutive_failures = 2
        _sync(provider, messages=_turn(*self._CALLS))
        self.assertEqual(len(backend.calls), 6)
        self.assertEqual(provider._consecutive_failures, 0)


@pytest.fixture
def buffer(tmp_path):
    return write_buffer.WriteBuffer(tmp_path / "pending.json")


def _bufferable():
    return CogneeUnreachable("refused", not_connected=True)


def test_bufferable_errors():
    assert write_buffer.is_bufferable_write_error(_bufferable())
    assert write_buffer.is_bufferable_write_error(CogneeHttpError(503, "warming up"))
    assert not write_buffer.is_bufferable_write_error(CogneeHttpError(500, "boom"))
    assert not write_buffer.is_bufferable_write_error(CogneeUnreachable("timed out"))
    assert not write_buffer.is_bufferable_write_error(RuntimeError("x"))


def test_never_connected_classification():
    assert _never_connected(urllib.error.URLError(ConnectionRefusedError()))
    assert _never_connected(urllib.error.URLError(socket.gaierror(8, "nodename")))
    assert not _never_connected(urllib.error.URLError(socket.timeout("timed out")))
    assert not _never_connected(TimeoutError("timed out"))
    assert not _never_connected(ConnectionResetError())


def test_drain_replays_oldest_first_and_empties_the_file(buffer):
    for n in range(3):
        buffer.enqueue(dataset="d", session_id="s", kind="qa", entry={"n": n})
    sent = []
    assert buffer.drain(lambda item, _timeout: sent.append(item["entry"]["n"]), budget_secs=5) == 3
    assert sent == [0, 1, 2]
    assert buffer.size() == 0
    assert not buffer.maybe_non_empty


def test_drain_stops_when_the_server_is_still_away(buffer):
    for n in range(3):
        buffer.enqueue(dataset="d", session_id="s", kind="qa", entry={"n": n})

    def send(item, _timeout=None):
        raise _bufferable()

    assert buffer.drain(send, budget_secs=5) == 0
    assert buffer.size() == 3


def test_drain_drops_an_entry_that_can_never_succeed(buffer):
    buffer.enqueue(dataset="d", session_id="s", kind="qa", entry={"n": 0})
    buffer.enqueue(dataset="d", session_id="s", kind="qa", entry={"n": 1})
    dropped = []

    def send(item, _timeout=None):
        if item["entry"]["n"] == 0:
            raise CogneeHttpError(400, "bad")

    stored = buffer.drain(send, budget_secs=5, on_drop=lambda item, exc: dropped.append(item))
    assert stored == 1
    assert [d["entry"]["n"] for d in dropped] == [0]
    assert buffer.size() == 0


def test_a_failed_send_puts_the_entry_back_in_its_place(buffer):
    for n in range(3):
        buffer.enqueue(dataset="d", session_id="s", kind="qa", entry={"n": n})
    sent = []

    def send(item, _timeout=None):
        if item["entry"]["n"] == 1:
            raise _bufferable()
        sent.append(item["entry"]["n"])

    assert buffer.drain(send, budget_secs=5) == 1
    assert sent == [0]
    data = json.loads(buffer.path.read_text())
    assert [e["entry"]["n"] for e in data["entries"]] == [1, 2]
    assert buffer.maybe_non_empty


def test_two_processes_draining_at_once_send_each_entry_exactly_once(tmp_path):
    # Two WriteBuffer instances on one file stand in for two Hermes processes:
    # each has its own drain lock, and only the file lock is shared.
    import threading
    from collections import Counter

    path = tmp_path / "pending.json"
    seed = write_buffer.WriteBuffer(path)
    for n in range(40):
        seed.enqueue(dataset="d", session_id="s", kind="qa", entry={"n": n})
    sent = Counter()
    lock = threading.Lock()
    start = threading.Barrier(2)

    def send(item, _timeout=None):
        time.sleep(0.005)  # a slow server, so the two drains overlap
        with lock:
            sent[item["entry"]["n"]] += 1

    def drain():
        buffer = write_buffer.WriteBuffer(path)
        start.wait()
        buffer.drain(send, budget_secs=10)

    workers = [threading.Thread(target=drain) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=20)
    assert sorted(sent) == list(range(40))
    assert set(sent.values()) == {1}, {n: c for n, c in sent.items() if c != 1}
    assert seed.size() == 0


def test_each_replayed_send_gets_what_is_left_of_the_budget(buffer):
    for n in range(6):
        buffer.enqueue(dataset="d", session_id="s", kind="qa", entry={"n": n})
    timeouts = []

    def send(item, timeout):
        timeouts.append(timeout)
        time.sleep(0.15)

    buffer.drain(send, budget_secs=0.5)
    assert timeouts[0] <= 0.5
    assert all(b < a for a, b in zip(timeouts, timeouts[1:])), timeouts
    # The budget bounds the drain: what it could not reach stays queued.
    assert buffer.size() == 6 - len(timeouts) > 0


def test_a_session_drain_waits_only_briefly_for_a_running_one(buffer, monkeypatch):
    import threading

    monkeypatch.setattr(write_buffer, "SESSION_DRAIN_LOCK_WAIT_SECS", 0.2)
    buffer.enqueue(dataset="d", session_id="s", kind="qa", entry={})
    release = threading.Event()

    def stuck(item, timeout):
        release.wait(5)

    runner = threading.Thread(target=buffer.drain, args=(stuck,), kwargs={"budget_secs": 5})
    runner.start()
    time.sleep(0.05)
    started = time.monotonic()
    assert buffer.drain(lambda item, _timeout: None, budget_secs=5, session_id="s") == 0
    assert time.monotonic() - started < 1.0
    release.set()
    runner.join(timeout=5)


def test_the_replay_runs_off_the_turns_thread():
    import threading

    backend = _ServerBackend()
    backend.down = False
    provider = make_provider(backend=backend)
    write_buffer.write_buffer_for(provider._write_buffer_path).enqueue(
        dataset="d", session_id="old", kind="qa", entry={"type": "qa"}
    )
    seen = {}
    gate = threading.Event()

    def slow_drain(budget_secs, session_id=""):
        seen["thread"] = threading.current_thread().name
        gate.wait(5)
        return 0

    provider._drain_write_buffer = slow_drain
    provider.sync_turn("q", "a")
    provider._sync_thread.join(timeout=1.0)
    # The turn's thread is done while the replay is still running.
    assert not provider._sync_thread.is_alive()
    gate.set()
    _settle_replays()
    assert seen["thread"] == "cognee-hermes-replay"


def test_drain_for_one_session_leaves_the_others(buffer):
    buffer.enqueue(dataset="d", session_id="a", kind="qa", entry={})
    buffer.enqueue(dataset="d", session_id="b", kind="qa", entry={})
    assert buffer.drain(lambda item, _timeout: None, budget_secs=5, session_id="a") == 1
    assert buffer.size("a") == 0
    assert buffer.size("b") == 1


def test_the_buffer_keeps_the_newest_week(buffer, monkeypatch):
    monkeypatch.setattr(write_buffer, "MAX_BUFFERED_ENTRIES", 3)
    for n in range(5):
        buffer.enqueue(dataset="d", session_id="s", kind="qa", entry={"n": n})
    data = json.loads(buffer.path.read_text())
    assert [e["entry"]["n"] for e in data["entries"]] == [2, 3, 4]
    stale = time.time() - write_buffer.MAX_BUFFERED_AGE_SECS - 10
    data["entries"][0]["queued_at"] = stale
    buffer.path.write_text(json.dumps(data))
    sent = []
    buffer.drain(lambda item, _timeout: sent.append(item["entry"]["n"]), budget_secs=5)
    assert sent == [3, 4]


if __name__ == "__main__":
    unittest.main(verbosity=2)
