"""One bounded, best-effort batching worker per process (SDK-1082 / SDK-1168).

See docs/telemetry-export-limits.md for memory accounting, loss semantics,
cloud event selection, local budgets and deployment prerequisites.
"""

import atexit
import os
import threading
import time
import weakref
from collections import deque
from contextlib import contextmanager
from typing import NamedTuple

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
from cognee.shared.telemetry_limits import (
    CLOUD_EVENTS,
    MILESTONE_EVENTS,
    PayloadTooLarge,
    TelemetryLimits,
    TokenBucket,
    bounded_json,
    reserve,
)

logger = get_logger()
BATCH_SIZE = 100
FLUSH_INTERVAL_SECONDS = 2.0
QUEUE_CAP = 1_000
RETRY_ATTEMPTS = 4
PREPARATION_CONCURRENCY = 4
ADMISSION_WAIT_SECONDS = 0.025
_COUNTERS = (
    "accepted_events",
    "exported_events",
    "exported_bytes",
    "sampled_events",
    "rejected_size",
    "rejected_format",
    "dropped_queue",
    "dropped_rate",
    "dropped_shutdown",
    "dropped_network",
    "dropped_http",
    "dropped_internal",
    "dropped_busy",
    "requests",
    "request_bytes",
    "retries",
    "legacy_fallbacks",
)


class _QueuedEvent(NamedTuple):
    body: bytes
    milestone: bool


class TelemetrySender:
    """Bounded admission, serialized byte queue, one network worker, bounded retries."""

    def __init__(
        self,
        url: str,
        request_timeout: float,
        transport: httpx.BaseTransport | None = None,
        *,
        limits: TelemetryLimits | None = None,
    ):
        self._url = url
        self._request_timeout = request_timeout
        self._transport = transport
        self.limits = limits or TelemetryLimits.from_env()
        self._reset()
        _senders.add(self)

    def count(self, category: str, amount: int = 1) -> None:
        # Fixed categories and saturating integers: no payloads or cardinality growth.
        with self._stats_lock:
            if category in self._counters:
                self._counters[category] = min(2**63 - 1, self._counters[category] + amount)

    def snapshot(self) -> dict[str, int]:
        """Local diagnostics for existing monitoring; does not emit another event."""
        with self._condition, self._stats_lock:
            return dict(
                self._counters,
                queue_events=len(self._queue),
                queue_bytes=self._queue_bytes,
                inflight_bytes=self._inflight_bytes,
                peak_queue_bytes=self._peak_queue_bytes,
                peak_inflight_bytes=self._peak_inflight_bytes,
            )

    @contextmanager
    def prepare(self, event_name: str, origin: str, properties: dict):
        """Bound concurrent payload preparation before sanitization/identity work."""
        if (origin == "cloud" or origin.startswith("cloud-")) and event_name not in CLOUD_EVENTS:
            self.count("sampled_events")
            yield False
            return
        if not self._preparation.acquire(timeout=ADMISSION_WAIT_SECONDS):
            self.count("dropped_busy")
            yield False
            return
        try:
            if self._closing.is_set():
                self.count("dropped_shutdown")
                yield False
                return
            try:
                # Permit bounded raw values that sanitization shrinks to fingerprints.
                # The complete encoded event still has the smaller wire-size cap.
                bounded_json(properties, 4 * self.limits.event_bytes)
            except PayloadTooLarge:
                self.count("rejected_size")
                yield False
                return
            except (ValueError, TypeError, RuntimeError):
                self.count("rejected_format")
                yield False
                return
            yield True
        finally:
            self._preparation.release()

    def enqueue(self, payload: dict) -> None:
        """Snapshot a payload; reserve bounded capacity for run milestones."""
        if not self._admission.acquire(timeout=ADMISSION_WAIT_SECONDS):
            self.count("dropped_busy")
            return
        try:
            name = payload.get("event_name") if type(payload) is dict else None
            milestone = type(name) is str and len(name) <= 256 and name in MILESTONE_EVENTS
            event_reserve = 0 if milestone else self._event_reserve
            byte_reserve = 0 if milestone else self._byte_reserve
            if self._closing.is_set():
                self.count("dropped_shutdown")
                return
            # Avoid serialization work once the event budget is exhausted.
            with self._condition:
                if not self._events.available(1 + event_reserve, time.monotonic()):
                    self.count("dropped_rate")
                    return
            try:
                encoded = bounded_json(payload, self.limits.event_bytes)
            except PayloadTooLarge:
                self.count("rejected_size")
                return
            except (ValueError, TypeError, RuntimeError):
                self.count("rejected_format")
                return
            size = len(encoded)
            # Worker and admission share only short bookkeeping sections; neither
            # serialization nor network IO holds the queue lock.
            with self._condition:
                if self._closing.is_set():
                    self.count("dropped_shutdown")
                    return
                now = time.monotonic()
                if not (
                    self._events.available(1 + event_reserve, now)
                    and self._bytes.available(size + byte_reserve, now)
                ):
                    self.count("dropped_rate")
                    return
                if milestone:
                    self._make_room_for_milestone(size)
                if self._queue_full(size):
                    self.count("dropped_queue")
                    return
                self._events.take(1)
                self._bytes.take(size)
                self._queue.append(_QueuedEvent(encoded, milestone))
                self._queue_bytes += size
                self._peak_queue_bytes = max(self._peak_queue_bytes, self._queue_bytes)
                self.count("accepted_events")
                try:
                    self._ensure_thread()
                except RuntimeError:
                    self._thread = None
                    self._drop_queue("dropped_internal")
                    return
                self._condition.notify()
        finally:
            self._admission.release()

    def close(self) -> None:
        """Flush best effort within one timeout; drop leftovers and forbid late requests.

        An already running request may finish later. Until it does, admission is
        closed and its single worker/bounded payload remain owned by this sender.
        """
        deadline = time.monotonic() + self._request_timeout
        with self._condition:
            thread = self._thread
            if thread is None:
                return
            self._deadline = min(self._deadline, deadline)
            self._closing.set()
            self._condition.notify()
        thread.join(max(0, deadline - time.monotonic()))
        with self._condition:
            if thread.is_alive():
                self._drop_queue("dropped_shutdown")

    def _reset(self) -> None:
        self._condition = threading.Condition()
        self._preparation = threading.BoundedSemaphore(PREPARATION_CONCURRENCY)
        self._admission = threading.BoundedSemaphore(PREPARATION_CONCURRENCY)
        self._stats_lock = threading.Lock()
        self._counters = dict.fromkeys(_COUNTERS, 0)
        self._queue: deque[_QueuedEvent] = deque()
        self._queue_bytes = self._inflight_bytes = 0
        self._peak_queue_bytes = self._peak_inflight_bytes = 0
        self._thread: threading.Thread | None = None
        self._closing = threading.Event()
        self._deadline = float("inf")
        # Explicit operator opt-in for an old proxy; HTTP 4xx alone never enables it.
        self._batches_supported = os.getenv("TELEMETRY_LEGACY_PROXY", "").lower() != "true"
        now = time.monotonic()
        self._events = TokenBucket(self.limits.events_per_minute, now)
        self._bytes = TokenBucket(self.limits.bytes_per_minute, now)
        self._wire_bytes = TokenBucket(self.limits.bytes_per_minute, now)
        self._requests = TokenBucket(self.limits.requests_per_minute, now)
        self._event_reserve = reserve(self.limits.events_per_minute)
        self._byte_reserve = reserve(
            self.limits.bytes_per_minute, 2 * (self.limits.event_bytes + 2)
        )
        self._request_reserve = reserve(self.limits.requests_per_minute)
        self._last_failure: int | str | None = None

    def _queue_full(self, size: int) -> bool:
        return len(self._queue) >= QUEUE_CAP or self._queue_bytes + size > self.limits.queue_bytes

    def _make_room_for_milestone(self, size: int) -> None:
        # One bounded pass, newest detail first; preserve FIFO among survivors.
        if not self._queue_full(size):
            return
        kept = deque()
        count = len(self._queue)
        for event in reversed(self._queue):
            full = count >= QUEUE_CAP or self._queue_bytes + size > self.limits.queue_bytes
            if full and not event.milestone:
                self._queue_bytes -= len(event.body)
                count -= 1
                self.count("dropped_queue")
            else:
                kept.appendleft(event)
        self._queue = kept

    def _ensure_thread(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="cognee-telemetry", daemon=True)
            self._thread.start()

    def _drop_queue(self, category: str) -> None:
        self.count(category, len(self._queue))
        self._queue.clear()
        self._queue_bytes = 0

    def _run(self) -> None:
        try:
            with httpx.Client(
                timeout=self._request_timeout,
                transport=self._transport,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            ) as client:
                while (batch := self._next_batch()) is not None:
                    try:
                        answered = self._send_batch(client, batch)
                    except Exception:
                        logger.debug(
                            "Telemetry send failed",
                            exc_info=(
                                RuntimeError,
                                RuntimeError("redacted telemetry failure"),
                                None,
                            ),
                        )
                        # Deliberately never log event data, responses, URLs or exception text.
                        self.count("dropped_internal", len(batch))
                        self._last_failure = "internal"
                        answered = False
                    finally:
                        with self._condition:
                            self._inflight_bytes = 0
                    # Release the previous batch before taking another one.
                    del batch
                    if not answered and self._closing.is_set() and self._last_failure != "budget":
                        with self._condition:
                            self._drop_queue("dropped_shutdown")
        except Exception:
            logger.debug(
                "Telemetry worker failed",
                exc_info=(RuntimeError, RuntimeError("redacted telemetry failure"), None),
            )
            with self._condition:
                self._drop_queue("dropped_internal")
        finally:
            with self._condition:
                self._inflight_bytes = 0
                self._thread = None
                self._closing.clear()
                self._deadline = float("inf")

    def _next_batch(self) -> list[_QueuedEvent] | None:
        with self._condition:
            while not self._queue and not self._closing.is_set():
                self._condition.wait()
            flush_at = time.monotonic() + FLUSH_INTERVAL_SECONDS
            while (
                len(self._queue) < BATCH_SIZE
                and self._queue_bytes + len(self._queue) + 1 < self.limits.batch_bytes
                and not self._closing.is_set()
            ):
                remaining = flush_at - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if time.monotonic() >= self._deadline:
                self._drop_queue("dropped_shutdown")
            if not self._queue:
                return None
            batch = []
            wire_size = 2
            while self._queue and len(batch) < BATCH_SIZE:
                event = self._queue[0]
                # Homogeneous batches prevent detail events spending the reserved
                # wire/request budget by sharing a batch with a run milestone.
                if batch and event.milestone != batch[0].milestone:
                    break
                size = len(event.body)
                next_size = wire_size + size + bool(batch)
                if next_size > self.limits.batch_bytes:
                    break
                batch.append(self._queue.popleft())
                self._queue_bytes -= size
                wire_size = next_size
            # Original encoded events + the contiguous HTTP body (including framing).
            self._inflight_bytes = sum(len(event.body) for event in batch) + wire_size
            self._peak_inflight_bytes = max(self._peak_inflight_bytes, self._inflight_bytes)
            return batch

    def _send_batch(self, client: httpx.Client, batch: list[_QueuedEvent]) -> bool:
        if self._batches_supported:
            parts = [b"["]
            for index, event in enumerate(batch):
                if index:
                    parts.append(b",")
                parts.append(event.body)
            parts.append(b"]")
            body = b"".join(parts)
            status = self._post(client, body, batch[0].milestone)
            del body, parts
            if status != "legacy":
                return self._record_result(status, batch)
            self._batches_supported = False
            self.count("legacy_fallbacks")
        for index, payload in enumerate(batch):
            status = self._post(client, payload.body, payload.milestone)
            if not self._record_result(status, [payload]):
                self.count(
                    _drop_category(status),
                    len(batch) - index - 1,
                )
                return False
        return True

    def _record_result(self, status: int | str | None, batch: list[_QueuedEvent]) -> bool:
        if isinstance(status, int) and 200 <= status < 300:
            self.count("exported_events", len(batch))
            self.count("exported_bytes", sum(len(event.body) for event in batch))
            return True
        self._last_failure = status
        self.count(_drop_category(status), len(batch))
        return False

    def _post(self, client: httpx.Client, body: bytes, milestone: bool) -> int | str | None:
        retrying = Retrying(
            stop=stop_after_attempt(RETRY_ATTEMPTS) | stop_when_event_set(self._closing),
            wait=wait_exponential_jitter(initial=1, max=4),
            sleep=self._sleep,
            retry=retry_if_result(
                lambda status: status is None or (isinstance(status, int) and status >= 500)
            ),
            retry_error_callback=_last_status,
        )
        return retrying(self._post_once, client, body, milestone)

    def _sleep(self, seconds: float) -> None:
        self.count("retries")
        self._closing.wait(seconds)

    def _post_once(self, client: httpx.Client, body: bytes, milestone: bool) -> int | str | None:
        now = time.monotonic()
        with self._condition:
            if now >= self._deadline:
                return "shutdown"
            if not (
                self._requests.available(1 + (0 if milestone else self._request_reserve), now)
                and self._wire_bytes.available(
                    len(body) + (0 if milestone else self._byte_reserve), now
                )
            ):
                return "budget"
            self._requests.take(1)
            self._wire_bytes.take(len(body))
        self.count("requests")
        self.count("request_bytes", len(body))
        try:
            # Do not buffer an unbounded response body from a broken proxy.
            with client.stream(
                "POST",
                self._url,
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=min(self._request_timeout, self._deadline - now),
            ) as response:
                if (
                    response.status_code == 422
                    and response.headers.get("X-Cognee-Telemetry-Batch") == "unsupported"
                ):
                    return "legacy"
                return response.status_code
        except httpx.HTTPError:
            return None


def _last_status(state: RetryCallState) -> int | str | None:
    return state.outcome.result() if state.outcome else None


def _drop_category(status: int | str | None) -> str:
    return {
        None: "dropped_network",
        "budget": "dropped_rate",
        "shutdown": "dropped_shutdown",
    }.get(status, "dropped_http")


# One pair of process hooks, not one pair retaining every constructed sender.
_senders: weakref.WeakSet[TelemetrySender] = weakref.WeakSet()


def _close_senders() -> None:
    for sender in list(_senders):
        sender.close()


def _reset_senders_after_fork() -> None:
    for sender in list(_senders):
        sender._reset()


atexit.register(_close_senders)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_senders_after_fork)
