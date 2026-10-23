"""Batched telemetry delivery: one proxy request per batch of events, not per event.

``send_telemetry`` hands each payload to ``TelemetrySender.enqueue``; a daemon
thread posts queued payloads as a JSON list once ``BATCH_SIZE`` events are
waiting or ``FLUSH_INTERVAL_SECONDS`` after the first one arrived. Best-effort
throughout: nothing here raises into the caller, and a full queue drops its
oldest events.
"""

import atexit
import os
import threading
import time
from collections import deque

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_result,
    stop_after_attempt,
    stop_when_event_set,
    wait_exponential_jitter,
)

from cognee.shared.logging_utils import get_logger

logger = get_logger()

BATCH_SIZE = 100
FLUSH_INTERVAL_SECONDS = 2.0
# Bounds memory while the proxy is unreachable; the oldest events go first.
QUEUE_CAP = 10_000
# One send plus up to three retries, for network errors and 5xx only.
RETRY_ATTEMPTS = 4


class TelemetrySender:
    """Queue telemetry payloads and post them in batches from one daemon thread."""

    def __init__(
        self, url: str, request_timeout: float, transport: httpx.BaseTransport | None = None
    ):
        self._url = url
        self._request_timeout = request_timeout
        # Tests pass an httpx.MockTransport; production uses the default network one.
        self._transport = transport
        self._reset()
        # A forked child inherits the queue and lock but not the thread.
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._reset)
        atexit.register(self.close)

    def enqueue(self, payload: dict) -> None:
        """Queue ``payload`` for the next batch; a full queue drops its oldest event."""
        with self._condition:
            self._queue.append(payload)
            self._ensure_thread()
            # Wake the thread to start the flush timer, or to send a full batch.
            if len(self._queue) == 1 or len(self._queue) >= BATCH_SIZE:
                self._condition.notify()

    def close(self) -> None:
        """Send what is queued, without retries, waiting at most one request timeout."""
        with self._condition:
            thread = self._thread
            if thread is None:
                return
            self._closing.set()
            self._condition.notify()
        thread.join(self._request_timeout)

    def _reset(self) -> None:
        # A new lock too: a forked child may inherit the parent's lock while held.
        self._condition = threading.Condition()
        self._queue: deque[dict] = deque(maxlen=QUEUE_CAP)
        self._thread: threading.Thread | None = None
        self._closing = threading.Event()
        self._batches_supported = True

    def _ensure_thread(self) -> None:
        # Daemon: the thread never keeps the process alive; close() flushes at exit.
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="cognee-telemetry", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        with httpx.Client(timeout=self._request_timeout, transport=self._transport) as client:
            while (batch := self._next_batch()) is not None:
                try:
                    answered = self._send_batch(client, batch)
                except Exception:
                    logger.debug("Ignoring exception in TelemetrySender._run", exc_info=True)
                    answered = False
                if not answered and self._closing.is_set():
                    # The proxy is failing: drop the rest rather than delay exit.
                    with self._condition:
                        self._queue.clear()

    def _next_batch(self) -> list[dict] | None:
        """Block until a batch is due and take it; None tells the thread to exit."""
        with self._condition:
            self._wait_for_batch()
            if not self._queue:
                # Closing with nothing left; the next enqueue starts a new thread.
                self._thread = None
                self._closing.clear()
                return None
            count = min(BATCH_SIZE, len(self._queue))
            return [self._queue.popleft() for _ in range(count)]

    def _wait_for_batch(self) -> None:
        """Wait (lock held) for a full batch, the flush interval, or close."""
        while not self._queue and not self._closing.is_set():
            self._condition.wait()
        deadline = time.monotonic() + FLUSH_INTERVAL_SECONDS
        while len(self._queue) < BATCH_SIZE and not self._closing.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._condition.wait(remaining)

    def _send_batch(self, client: httpx.Client, batch: list[dict]) -> bool:
        """Post ``batch``; return False as soon as the proxy does not answer."""
        if self._batches_supported:
            status = self._post(client, batch)
            if status is None or not 400 <= status < 500:
                return _answered(status)
            # A proxy without batch support rejects the list: send events one by one.
            self._batches_supported = False
        for payload in batch:
            if not _answered(self._post(client, payload)):
                return False
        return True

    def _post(self, client: httpx.Client, body: dict | list[dict]) -> int | None:
        """POST ``body``, retrying 5xx and network errors with backoff; return the last status."""
        retrying = Retrying(
            # No retries once closing: exit must not wait on a failing proxy.
            stop=stop_after_attempt(RETRY_ATTEMPTS) | stop_when_event_set(self._closing),
            # Jitter spreads retries from many processes after a proxy outage.
            wait=wait_exponential_jitter(initial=1, max=4),
            sleep=self._sleep,
            retry=retry_if_result(lambda status: not _answered(status)),
            retry_error_callback=_last_status,
        )
        return retrying(self._post_once, client, body)

    def _sleep(self, seconds: float) -> None:
        # Backoff that close() cuts short instead of waiting it out.
        self._closing.wait(seconds)

    def _post_once(self, client: httpx.Client, body: dict | list[dict]) -> int | None:
        try:
            response = client.post(self._url, json=body)
        except httpx.HTTPError as error:
            logger.debug("Telemetry request failed: %s", error)
            return None
        if response.status_code != 200:
            logger.debug("Telemetry proxy returned status %s", response.status_code)
        return response.status_code


def _answered(status: int | None) -> bool:
    """Whether the proxy handled the request: no network error, timeout, or 5xx."""
    return status is not None and status < 500


def _last_status(state: RetryCallState) -> int | None:
    """Out of attempts: return the last status instead of raising RetryError."""
    return state.outcome.result() if state.outcome else None
