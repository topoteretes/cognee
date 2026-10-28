"""Capture sinks (SDK-529).

A sink is a bare async callable taking a list of already-serialized records:
``async def sink(records: list[dict]) -> None``. Sinks are invoked only from
the background flusher or from ``drain()``/``shutdown()``, never inline from
``emit()``. Two ship: ``StorageSink`` (gzip'd JSONL under COGNEE_CAPTURE_DIR,
here) and ``ProxySink`` (the analytics proxy, ``proxy_sink.py``);
``FanOutSink`` runs several. A sink must never mutate the records it is handed
— a fan-out hands the same list to every sink.

Contract for implementations: usable from any loop/thread — two flushers on
different loops (run_sync's thread, the dataset-queue reaper thread) can call a
sink concurrently — so no loop-bound clients. ``StorageManager`` Local/S3
qualify. Delivery is at-least-once: a flush cancelled or cut off by
``drain()``'s budget mid-write is re-buffered and written again later, so a
sink may receive the same records twice (``StorageSink`` never overwrites —
blob names are collision-free — so consumers that need exactly-once counts
dedupe on the record's ``event_id``, which is the same on both copies). Sinks
must raise ``Exception``
subclasses only: a ``BaseException`` from a sink re-buffers the batch and ends
the flusher (the next emit starts a replacement). ``KeyboardInterrupt`` and
``SystemExit`` are the exception to that: asyncio's ``Task.__step`` re-raises
both out of the event loop, so they escape past the flusher and cannot be
contained here. A sink must let ``CancelledError`` propagate — the flusher
cancels a write that outlives ``SINK_TIMEOUT_S`` or a ``drain()`` budget and
waits ``_CANCEL_GRACE_S`` for that cancel to land before abandoning the write
and re-buffering the batch.

A sink that wrote SOME of the records it was handed — the proxy sink leaves
out a record it cannot serialize or that is over the Firehose record limit,
and a request that fails leaves the later ones unsent — raises
``PartialDelivery`` AFTER writing what it could, naming the positions it did
not deliver. The flusher credits the rest to their runs and charges those
positions as dropped; a plain exception charges the whole batch.

Off-loop work — serialization, gzip, every sink write — runs on the capture
pool (``_CapturePool``: two worker threads of its own), never on the loop's
default executor: capture must not queue behind, or take workers from, the
``to_thread`` work of the pipeline it observes, and a wedged write abandoned
by a timeout keeps only a capture worker busy. The pool is small on purpose and
does not queue: with both workers busy a submission raises
``CapturePoolSaturated`` and the flusher counts that batch as dropped, instead
of lining batches up behind writes that may never return. Sinks must not
depend on the default executor or on threads of their own: the atexit drain
runs after ``threading._shutdown()``, when ``asyncio.to_thread`` raises and the
write is lost; ``StorageSink`` goes through ``run_off_loop`` /
``run_coroutine_off_loop``, which fall back to an inline call there.

A sink write must have a REAL suspension point. The flusher bounds each write
with ``SINK_TIMEOUT_S`` (and ``drain()`` with its budget) by cancelling it, and
a coroutine can only be cancelled where it awaits something that actually
suspends: ``LocalFileStorage.store`` is an ``async def`` whose body never does
(makedirs, write, rename), so awaited directly it would pin the flusher — and
every other coroutine on that loop — for as long as the filesystem takes, and
neither timeout could fire. ``StorageSink`` therefore drives the whole write on
a worker thread (``run_coroutine_off_loop``).
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import gzip
import io
import itertools
import json
import os
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar

from cognee.shared.logging_utils import get_logger

from .events import KIND_RUN_MANIFEST

if TYPE_CHECKING:
    from cognee.infrastructure.files.storage import StorageManager

logger = get_logger("eval_capture")

CaptureSink = Callable[[list[dict]], Awaitable[None]]

T = TypeVar("T")

# Worker threads of the capture pool. Two: one flusher's serialization or write
# at a time, plus one for a second loop's flusher (run_sync's thread, the
# dataset-queue reaper) or for a write a timeout abandoned.
CAPTURE_POOL_WORKERS = 2


class CapturePoolSaturated(Exception):
    """Both capture workers are busy: the submission was refused, not queued.

    Deliberately not a ``RuntimeError``: ``run_off_loop`` falls back to an
    inline call on that (the pool gone at interpreter exit), and a saturated
    pool must drop the batch instead, so a wedged sink cannot move its wait
    onto the event loop.
    """


class PartialDelivery(Exception):
    """Raised by a sink that delivered some of its records and not others.

    ``undelivered`` holds the positions, in the list the sink was handed, of the
    records that did not reach the destination; the sink raises it AFTER
    writing what it could. The flusher credits every other record to its run
    and charges these as dropped, so ``events_delivered`` and
    ``capture_complete`` describe what actually landed. A sink that delivered
    nothing raises its own error, which charges the whole batch.
    """

    def __init__(self, undelivered: Iterable[int], reason: str | None = None) -> None:
        self.undelivered: tuple[int, ...] = tuple(sorted(set(undelivered)))
        super().__init__(reason or f"{len(self.undelivered)} record(s) not delivered")


class _CapturePool:
    """The capture-only thread pool; see the module docstring.

    ``busy`` counts submissions not yet finished — including a worker still
    running a function the awaiting side gave up on (a cancelled future never
    interrupts a thread) — so saturation means "both workers are occupied",
    whether by live work or by abandoned writes. The executor is created on
    first use, so importing capture starts no threads.
    """

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._busy = 0
        self._lock = threading.Lock()

    @property
    def busy(self) -> int:
        return self._busy

    def submit(
        self, loop: asyncio.AbstractEventLoop, func: Callable[..., T], /, *args: Any
    ) -> asyncio.Future[T]:
        """Schedule ``func`` on a capture worker; returns a future bound to ``loop``.

        Raises ``CapturePoolSaturated`` when both workers are busy, and the
        executor's ``RuntimeError`` once the interpreter is shutting down —
        callers fall back to an inline call on that one only.
        """
        with self._lock:
            if self._busy >= self.max_workers:
                raise CapturePoolSaturated(
                    f"{self._busy} capture worker(s) busy, submission refused"
                )
            executor = self._executor
            if executor is None:
                executor = self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix="cognee-capture"
                )
            self._busy += 1
        try:
            future = executor.submit(func, *args)
        except BaseException:
            with self._lock:
                self._busy -= 1
            raise
        # Bound to this executor: after reset() a worker of the old executor
        # finishing late must not decrement the new count.
        future.add_done_callback(functools.partial(self._release, executor))
        return asyncio.wrap_future(future, loop=loop)

    def _release(self, executor: ThreadPoolExecutor, _future: ConcurrentFuture) -> None:
        with self._lock:
            if self._executor is executor:
                self._busy -= 1

    def reset(self) -> None:
        """Forget the executor; its threads finish their current work on their own."""
        with self._lock:
            executor, self._executor, self._busy = self._executor, None, 0
        if executor is not None:
            executor.shutdown(wait=False)


_pool = _CapturePool(CAPTURE_POOL_WORKERS)


def _reset_pool_for_tests() -> None:
    """A fresh pool per test, so a write one test abandoned cannot saturate the next."""
    _pool.reset()


async def run_off_loop(func: Callable[..., T], /, *args: Any) -> T:
    """Run pure-CPU work on a capture worker, inline once the pool is gone.

    Serialization and gzip run on the capture pool so they never freeze
    concurrent coroutines. Inside atexit handlers ``threading._shutdown()`` has
    already run and the pool refuses new work with ``RuntimeError`` ("cannot
    schedule new futures after interpreter shutdown") — raised at SUBMIT time,
    before ``func`` starts. Falling back to an inline call there is what lets
    the atexit drain persist the batch it popped instead of silently losing it.
    Shared by the flusher and the sinks.

    Only that submit-time failure triggers the fallback: an error raised by
    ``func`` itself in the worker thread — a ``RecursionError`` (a
    ``RuntimeError`` subclass) on a pathologically deep payload, say —
    propagates instead of being retried on the event loop, and so does
    ``CapturePoolSaturated`` (not a ``RuntimeError``), which the flusher
    accounts as a dropped batch.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    try:
        future = _pool.submit(loop, context.run, func, *args)
    except RuntimeError:
        return func(*args)
    return await future


def _run_to_completion(make_coroutine: Callable[[], Awaitable[T]]) -> T:
    """Worker-thread body: build the coroutine HERE (never on the caller's side,
    so a submit-time failure leaves no never-awaited coroutine behind) and drive
    it on a private loop."""
    return asyncio.run(make_coroutine())


async def run_coroutine_off_loop(make_coroutine: Callable[[], Awaitable[T]]) -> T:
    """Drive a coroutine to completion in a worker thread; inline once the executor is gone.

    For an ``async def`` whose body blocks — ``LocalFileStorage.store`` (no
    suspension point at all) — or that offloads internally with no
    interpreter-exit fallback (``S3FileStorage.store``). Running it on its own
    loop in a worker thread gives the awaiting flusher a real suspension point,
    so ``SINK_TIMEOUT_S`` and ``drain()``'s budget can actually cut the wait
    (the worker then finishes on its own: abandoned, not interrupted — which is
    why blob names are collision-free and manifests are written atomically) and
    a stalled filesystem no longer freezes every coroutine on the flusher's loop.
    Same submit-time fallback as ``run_off_loop``: inside the atexit drain the
    pool is gone, and the coroutine is awaited inline on the drain's own loop.
    """
    loop = asyncio.get_running_loop()
    try:
        future = _pool.submit(loop, _run_to_completion, make_coroutine)
    except RuntimeError:
        return await make_coroutine()
    return await future


# Process-wide blob sequence: together with time_ns and pid this makes blob
# names collision-free across loops/threads/processes without coordination. A
# per-sink NNNNNN counter with overwrite=True would silently clobber blobs
# under two concurrent flushers.
_blob_sequence = itertools.count()


def _encode_group(records: list[dict]) -> bytes:
    """JSONL + gzip. compresslevel=1 is ~2x faster than 6 for a marginal size cost."""
    lines = "".join(json.dumps(record, default=str) + "\n" for record in records)
    return gzip.compress(lines.encode("utf-8"), compresslevel=1)


class StorageSink:
    """Write capture records to a ``StorageManager`` (Local or S3).

    Layout under ``root``::

        {dataset or "nodataset"}/{run or "norun"}/{kind}/batch-{ts_ns}-{pid}-{seq:06d}.jsonl.gz
        {dataset or "nodataset"}/{run or "norun"}/manifest.json   (kind run.manifest, pretty JSON)

    Every file carries the same record envelope (``schema_version``,
    ``event_id``, ``kind``, ``run_id``, ``dataset_id``, ``user_id``,
    ``tenant_id``, ``stage``, ``ts``, ``payload``); the manifest's fields live
    under ``payload``. Nothing is written to the relational DB.

    Runs on the flusher task only, never on an emit path. Each write — the
    encode and the ``store()`` call — runs as one unit on a worker thread
    (``run_coroutine_off_loop``): ``LocalFileStorage.store`` is an ``async def``
    with a blocking body (makedirs, write, rename), sub-millisecond on local
    disk and unbounded on a slow network filesystem, and awaited on the loop it
    would pin the flusher past every timeout (see the module docstring).
    """

    def __init__(self, storage: StorageManager, root: str = "") -> None:
        self._storage = storage
        self._root = root if not root or root.endswith("/") else root + "/"

    async def __call__(self, records: list[dict]) -> None:
        groups: dict[tuple[str, str, str], list[dict]] = {}
        for record in records:
            key = (
                record.get("dataset_id") or "nodataset",
                record.get("run_id") or "norun",
                record["kind"],
            )
            groups.setdefault(key, []).append(record)

        for (dataset, run, kind), group in groups.items():
            if kind == KIND_RUN_MANIFEST:
                for record in group:
                    await run_coroutine_off_loop(
                        functools.partial(
                            self._write_manifest,
                            f"{self._root}{dataset}/{run}/manifest.json",
                            record,
                        )
                    )
                continue

            blob_name = f"batch-{time.time_ns()}-{os.getpid()}-{next(_blob_sequence):06d}.jsonl.gz"
            await run_coroutine_off_loop(
                functools.partial(
                    self._write_blob, f"{self._root}{dataset}/{run}/{kind}/{blob_name}", group
                )
            )

    async def _write_manifest(self, path: str, record: dict) -> None:
        # str branch writes UTF-8; overwrite is write-then-rename on Local.
        await self._storage.store(path, json.dumps(record, indent=2, default=str), overwrite=True)

    async def _write_blob(self, path: str, group: list[dict]) -> None:
        # Encode and write on the same worker: the encode is pure CPU, the write
        # blocks. BytesIO, not raw bytes, honors the ``BinaryIO | str`` annotation.
        await self._storage.store(path, io.BytesIO(_encode_group(group)), overwrite=True)


class FanOutSink:
    """Deliver every batch to several sinks (``COGNEE_CAPTURE_SINK=both``).

    Sinks run in order, each given the same records. One sink's failure does
    not starve the next: every sink runs, then the first exception is
    re-raised so the flusher accounts the batch as dropped. The at-least-once
    contract therefore holds per batch, not per sink — a batch one sink stored
    and another lost is counted lost on the run (``events_dropped``), which is
    the honest reading of "did the warehouse get it". A sink that delivered
    part of the batch (``PartialDelivery``) contributes the positions it left
    out: the batch's undelivered set is the union over sinks, and a sink that
    failed outright still fails the whole batch.
    """

    def __init__(self, sinks: Sequence[CaptureSink]) -> None:
        if not sinks:
            raise ValueError("FanOutSink needs at least one sink")
        self.sinks: tuple[CaptureSink, ...] = tuple(sinks)

    async def __call__(self, records: list[dict]) -> None:
        first_error: Exception | None = None
        undelivered: set[int] = set()
        for sink in self.sinks:
            try:
                await sink(records)
            except asyncio.CancelledError:
                raise
            except PartialDelivery as exc:
                undelivered.update(exc.undelivered)
                logger.debug("capture sink %r delivered part of a batch in fan-out (%s)", sink, exc)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logger.debug("capture sink %r failed in fan-out (%s)", sink, exc)
        if first_error is not None:
            raise first_error
        if undelivered:
            raise PartialDelivery(undelivered)
