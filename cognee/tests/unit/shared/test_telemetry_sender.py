"""Tests for batched telemetry delivery (SDK-1082): one proxy request per batch."""

import json
import time
from uuid import uuid4

import httpx
import pytest
from tenacity import wait_none

from cognee.shared import telemetry_sender
from cognee.shared.telemetry_sender import TelemetrySender


class FakeProxy:
    """Records request bodies; answers with scripted statuses or errors, then 200."""

    def __init__(self, *responses: int | Exception):
        self.bodies: list = []
        self._responses = list(responses)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        response = self._responses.pop(0) if self._responses else 200
        if isinstance(response, Exception):
            raise response
        return httpx.Response(response)


@pytest.fixture
def make_sender(monkeypatch):
    monkeypatch.setattr(telemetry_sender, "FLUSH_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(telemetry_sender, "wait_exponential_jitter", lambda **_: wait_none())
    senders: list[TelemetrySender] = []

    def make(proxy: FakeProxy, request_timeout: float = 1.0) -> TelemetrySender:
        transport = httpx.MockTransport(proxy.handle)
        senders.append(TelemetrySender("http://proxy.test", request_timeout, transport))
        return senders[-1]

    yield make
    for sender in senders:
        sender.close()


def _events(count: int, start: int = 0) -> list[dict]:
    return [{"n": n} for n in range(start, start + count)]


def _wait_until(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.01)


def test_full_batch_is_sent_as_one_request(make_sender, monkeypatch):
    monkeypatch.setattr(telemetry_sender, "FLUSH_INTERVAL_SECONDS", 10)
    proxy = FakeProxy()
    sender = make_sender(proxy)

    for event in _events(telemetry_sender.BATCH_SIZE):
        sender.enqueue(event)

    _wait_until(lambda: proxy.bodies)
    assert proxy.bodies == [_events(telemetry_sender.BATCH_SIZE)]


def test_partial_batch_is_sent_after_the_flush_interval(make_sender):
    proxy = FakeProxy()
    sender = make_sender(proxy)

    for event in _events(3):
        sender.enqueue(event)
    assert proxy.bodies == []

    _wait_until(lambda: proxy.bodies)
    assert proxy.bodies == [_events(3)]


def test_close_sends_queued_events_at_once(make_sender, monkeypatch):
    monkeypatch.setattr(telemetry_sender, "FLUSH_INTERVAL_SECONDS", 10)
    proxy = FakeProxy()
    sender = make_sender(proxy)
    for event in _events(3):
        sender.enqueue(event)

    started = time.monotonic()
    sender.close()

    assert time.monotonic() - started < 1
    assert proxy.bodies == [_events(3)]
    assert sender._thread is None


def test_sender_restarts_after_close(make_sender):
    proxy = FakeProxy()
    sender = make_sender(proxy)
    sender.enqueue({"n": 0})
    sender.close()

    sender.enqueue({"n": 1})
    sender.close()

    assert proxy.bodies == [[{"n": 0}], [{"n": 1}]]


def test_full_queue_drops_oldest_events(make_sender, monkeypatch):
    monkeypatch.setattr(telemetry_sender, "FLUSH_INTERVAL_SECONDS", 10)
    monkeypatch.setattr(telemetry_sender, "QUEUE_CAP", 5)
    proxy = FakeProxy()
    sender = make_sender(proxy)

    for event in _events(8):
        sender.enqueue(event)
    sender.close()

    assert proxy.bodies == [_events(5, start=3)]


def test_queue_over_its_byte_cap_drops_oldest_events(make_sender, monkeypatch):
    monkeypatch.setattr(telemetry_sender, "FLUSH_INTERVAL_SECONDS", 10)
    # Room for three '{"n":0}' events of 7 bytes each.
    monkeypatch.setattr(telemetry_sender, "QUEUE_BYTES", 21)
    proxy = FakeProxy()
    sender = make_sender(proxy)

    for event in _events(5):
        sender.enqueue(event)
    sender.close()

    assert proxy.bodies == [_events(3, start=2)]


def test_batches_stay_within_their_byte_cap(make_sender, monkeypatch):
    monkeypatch.setattr(telemetry_sender, "FLUSH_INTERVAL_SECONDS", 10)
    # '[' + two 7-byte events + ',' + ']': exactly two events per request body.
    monkeypatch.setattr(telemetry_sender, "BATCH_BYTES", 17)
    proxy = FakeProxy()
    sender = make_sender(proxy)

    for event in _events(5):
        sender.enqueue(event)
    sender.close()

    assert proxy.bodies == [_events(2), _events(2, start=2), _events(1, start=4)]


def test_event_larger_than_a_batch_is_sent_alone(make_sender, monkeypatch):
    # Smaller than one 7-byte event in brackets: no batch can hold even one.
    monkeypatch.setattr(telemetry_sender, "BATCH_BYTES", 5)
    proxy = FakeProxy()
    sender = make_sender(proxy)

    for event in _events(2):
        sender.enqueue(event)
    sender.close()

    assert proxy.bodies == [_events(1), _events(1, start=1)]


def test_bad_or_oversized_event_is_dropped_alone(make_sender, monkeypatch):
    monkeypatch.setattr(telemetry_sender, "FLUSH_INTERVAL_SECONDS", 10)
    proxy = FakeProxy()
    sender = make_sender(proxy)
    oversized = {"pad": "x" * telemetry_sender.EVENT_BYTES}

    for event in [{"n": 0}, {"id": uuid4()}, {"n": float("nan")}, oversized, {"n": 1}]:
        sender.enqueue(event)
    sender.close()

    assert proxy.bodies == [_events(2)]


def test_server_errors_and_network_errors_are_retried(make_sender):
    proxy = FakeProxy(503, httpx.ConnectError("proxy down"), 200)
    sender = make_sender(proxy)

    sender.enqueue({"n": 0})

    _wait_until(lambda: len(proxy.bodies) == 3)
    assert proxy.bodies == [[{"n": 0}]] * 3


def test_batch_is_dropped_after_the_last_retry(make_sender):
    proxy = FakeProxy(*[503] * telemetry_sender.RETRY_ATTEMPTS)
    sender = make_sender(proxy)

    sender.enqueue({"n": 0})
    _wait_until(lambda: len(proxy.bodies) == telemetry_sender.RETRY_ATTEMPTS)
    sender.enqueue({"n": 1})
    # Not close(): a failure still in flight while closing would drop event 1 too.
    _wait_until(lambda: len(proxy.bodies) == telemetry_sender.RETRY_ATTEMPTS + 1)

    assert proxy.bodies == [[{"n": 0}]] * telemetry_sender.RETRY_ATTEMPTS + [[{"n": 1}]]


def test_client_errors_drop_the_batch_without_retrying_or_splitting_it(make_sender):
    proxy = FakeProxy(422, 413, 429)
    sender = make_sender(proxy)

    for n in range(3):
        sender.enqueue({"n": n})
        _wait_until(lambda n=n: len(proxy.bodies) == n + 1)

    assert proxy.bodies == [[{"n": 0}], [{"n": 1}], [{"n": 2}]]


def test_close_gives_up_at_the_first_failure(make_sender, monkeypatch):
    monkeypatch.setattr(telemetry_sender, "BATCH_SIZE", 2)
    proxy = FakeProxy(503)
    sender = make_sender(proxy)
    # Three batches queued before the thread can run, so its first send is already closing.
    with sender._condition:
        for event in _events(6):
            sender.enqueue(event)
        sender._closing.set()

    sender.close()

    assert proxy.bodies == [_events(2)]


def test_close_waits_at_most_one_request_timeout(make_sender):
    class HangingProxy(FakeProxy):
        # MockTransport enforces no timeout: a hanging proxy, as far as close() can tell.
        def handle(self, request: httpx.Request) -> httpx.Response:
            time.sleep(1.5)
            return super().handle(request)

    sender = make_sender(HangingProxy(), request_timeout=0.2)
    sender.enqueue({"n": 0})

    started = time.monotonic()
    sender.close()

    assert time.monotonic() - started < 1
