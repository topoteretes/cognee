"""Capture sinks (SDK-529).

A sink is a bare async callable taking a list of already-serialized records
(shape-compatible with ``ActivitySink``). Sinks are invoked only from the
background flusher or from ``drain()``/``shutdown()``, never inline from
``emit()``.

Contract for implementations: usable from any loop/thread — two flushers on
different loops (run_sync's thread, the dataset-queue reaper thread) can call a
sink concurrently — so no loop-bound clients. ``StorageManager`` Local/S3
qualify. Delivery is at-least-once: a flush cancelled or cut off by
``drain()``'s budget mid-write is re-buffered and written again later, so a
sink may receive the same records twice (``StorageSink`` never overwrites —
blob names are collision-free — so consumers that need exactly-once counts
should dedupe on ``(run_id, kind, ts)``). Sinks must raise ``Exception``
subclasses only; a ``BaseException`` from a sink escapes the flusher's
never-raise guards by design.
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

    Every file carries the same record envelope (``kind``, ``run_id``,
    ``dataset_id``, ``stage``, ``ts``, ``payload``); the manifest's fields live
    under ``payload``. Nothing is written to the relational DB.
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
                    # str branch writes UTF-8; overwrite is write-then-rename on Local.
                    await self._storage.store(
                        f"{self._root}{dataset}/{run}/manifest.json",
                        json.dumps(record, indent=2, default=str),
                        overwrite=True,
                    )
                continue

            # Encoding is pure CPU; keep it off the loop like the flusher's
            # serialization step (inline at interpreter exit, see run_off_loop).
            blob = await run_off_loop(_encode_group, group)
            blob_name = f"batch-{time.time_ns()}-{os.getpid()}-{next(_blob_sequence):06d}.jsonl.gz"
            # BytesIO, not raw bytes, honors the ``BinaryIO | str`` annotation.
            await self._storage.store(
                f"{self._root}{dataset}/{run}/{kind}/{blob_name}",
                io.BytesIO(blob),
                overwrite=True,
            )

    write = __call__
