"""Capture sinks (SDK-529).

A sink is a bare async callable taking a list of already-serialized records:
``async def sink(records: list[dict]) -> None``. Sinks are invoked only from
the background flusher or from ``drain()``/``shutdown()``, never inline from
``emit()``.

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
Sinks must not depend on the default executor or on threads: the atexit drain
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
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

from .events import KIND_RUN_MANIFEST

if TYPE_CHECKING:
    from cognee.infrastructure.files.storage import StorageManager

CaptureSink = Callable[[list[dict]], Awaitable[None]]

T = TypeVar("T")


async def run_off_loop(func: Callable[..., T], /, *args: Any) -> T:
    """Run pure-CPU work in a worker thread, inline once the executor is gone.

    Serialization and gzip run in the default executor (the body of
    ``asyncio.to_thread``) so they never freeze concurrent coroutines. Inside
    atexit handlers ``threading._shutdown()`` has already run and the executor
    refuses new work with ``RuntimeError`` ("cannot schedule new futures after
    interpreter shutdown") — raised at SUBMIT time, before ``func`` starts.
    Falling back to an inline call there is what lets the atexit drain persist
    the batch it popped instead of silently losing it. Shared by the flusher
    and ``StorageSink``.

    Only the submit-time failure triggers the fallback: an error raised by
    ``func`` itself in the worker thread — a ``RecursionError`` (a
    ``RuntimeError`` subclass) on a pathologically deep payload, say —
    propagates instead of being retried on the event loop.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    try:
        future = loop.run_in_executor(None, functools.partial(context.run, func, *args))
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
    executor is gone, and the coroutine is awaited inline on the drain's own loop.
    """
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(None, functools.partial(_run_to_completion, make_coroutine))
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
    ``event_id``, ``kind``, ``run_id``, ``dataset_id``, ``stage``, ``ts``,
    ``payload``); the manifest's fields live under ``payload``. Nothing is
    written to the relational DB.

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
