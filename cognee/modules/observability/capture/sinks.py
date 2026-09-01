"""Capture sinks (SDK-529).

A sink is a bare async callable taking a list of already-serialized records
(shape-compatible with ``ActivitySink``). Sinks are invoked only from the
background flusher or from ``drain()``/``shutdown()``, never inline from
``emit()``.

Contract for implementations: usable from any loop/thread — two flushers on
different loops (run_sync's thread, the dataset-queue reaper thread) can call a
sink concurrently — so no loop-bound clients. ``StorageManager`` Local/S3
qualify.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import itertools
import json
import os
import time
from typing import TYPE_CHECKING, Awaitable, Callable

from .events import KIND_RUN_MANIFEST

if TYPE_CHECKING:
    from cognee.infrastructure.files.storage import StorageManager

CaptureSink = Callable[[list[dict]], Awaitable[None]]

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
            # serialization step.
            blob = await asyncio.to_thread(_encode_group, group)
            blob_name = f"batch-{time.time_ns()}-{os.getpid()}-{next(_blob_sequence):06d}.jsonl.gz"
            # BytesIO, not raw bytes, honors the ``BinaryIO | str`` annotation.
            await self._storage.store(
                f"{self._root}{dataset}/{run}/{kind}/{blob_name}",
                io.BytesIO(blob),
                overwrite=True,
            )

    write = __call__
