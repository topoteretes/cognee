"""StorageSink layout (SDK-529): gzip'd JSONL blobs per (dataset, run, kind) and a
pretty manifest.json per run, written through the normal StorageManager."""

import asyncio
import gzip
import json
import re
import time
from uuid import uuid4

import pytest

from cognee.infrastructure.files.storage import StorageManager
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.modules.observability import capture
from cognee.modules.observability.capture import (
    KIND_RUN_MANIFEST,
    KIND_SUMMARY_GENERATED,
    StorageSink,
    hook,
)

pytestmark = pytest.mark.usefixtures("capture_reset")


def _record(kind, run_id, dataset_id, payload, ts=1.0):
    return {
        "kind": kind,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "stage": None,
        "ts": ts,
        "payload": payload,
    }


def _read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as blob:
        return [json.loads(line) for line in blob if line.strip()]


@pytest.mark.asyncio
async def test_layout_roundtrip_manifest_and_nodataset(tmp_path):
    sink = StorageSink(StorageManager(LocalFileStorage(str(tmp_path))))
    run_id = str(uuid4())
    dataset_id = str(uuid4())

    await sink(
        [
            _record(KIND_SUMMARY_GENERATED, run_id, dataset_id, "s1", ts=1.0),
            _record(KIND_SUMMARY_GENERATED, run_id, dataset_id, "s2", ts=2.0),
            _record(KIND_RUN_MANIFEST, run_id, dataset_id, {"run_id": run_id, "kind": "pipeline"}),
            _record(KIND_SUMMARY_GENERATED, run_id, None, "orphan"),
            _record(KIND_SUMMARY_GENERATED, None, dataset_id, "runless"),
        ]
    )

    blobs = list((tmp_path / dataset_id / run_id / KIND_SUMMARY_GENERATED).glob("batch-*.jsonl.gz"))
    assert len(blobs) == 1
    # Collision-free shape: batch-{ts_ns}-{pid}-{seq:06d}.jsonl.gz
    assert re.fullmatch(r"batch-\d+-\d+-\d{6}\.jsonl\.gz", blobs[0].name), blobs[0].name
    lines = _read_jsonl_gz(blobs[0])
    assert [line["payload"] for line in lines] == ["s1", "s2"]
    assert lines[0]["kind"] == KIND_SUMMARY_GENERATED
    assert lines[0]["run_id"] == run_id

    manifest_path = tmp_path / dataset_id / run_id / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "\n  " in manifest_text  # pretty-printed
    manifest = json.loads(manifest_text)
    assert manifest["kind"] == KIND_RUN_MANIFEST
    assert manifest["payload"] == {"run_id": run_id, "kind": "pipeline"}

    orphan_blobs = list(
        (tmp_path / "nodataset" / run_id / KIND_SUMMARY_GENERATED).glob("batch-*.jsonl.gz")
    )
    assert len(orphan_blobs) == 1
    assert [line["payload"] for line in _read_jsonl_gz(orphan_blobs[0])] == ["orphan"]

    [runless_blob] = (tmp_path / dataset_id / "norun" / KIND_SUMMARY_GENERATED).glob("*.jsonl.gz")
    assert [line["payload"] for line in _read_jsonl_gz(runless_blob)] == ["runless"]


@pytest.mark.asyncio
async def test_successive_writes_never_clobber(tmp_path):
    sink = StorageSink(StorageManager(LocalFileStorage(str(tmp_path))))
    run_id = str(uuid4())

    await sink([_record(KIND_SUMMARY_GENERATED, run_id, None, "a")])
    await sink([_record(KIND_SUMMARY_GENERATED, run_id, None, "b")])

    blobs = sorted((tmp_path / "nodataset" / run_id / KIND_SUMMARY_GENERATED).glob("*.jsonl.gz"))
    assert len(blobs) == 2
    payloads = sorted(line["payload"] for blob in blobs for line in _read_jsonl_gz(blob))
    assert payloads == ["a", "b"]


@pytest.mark.asyncio
async def test_manifest_overwrite_keeps_latest_and_root_prefix(tmp_path):
    sink = StorageSink(StorageManager(LocalFileStorage(str(tmp_path))), root="evals-run-1")
    run_id = str(uuid4())
    dataset_id = str(uuid4())

    await sink([_record(KIND_RUN_MANIFEST, run_id, dataset_id, {"v": 1})])
    await sink([_record(KIND_RUN_MANIFEST, run_id, dataset_id, {"v": 2})])

    manifest_path = tmp_path / "evals-run-1" / dataset_id / run_id / "manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["payload"] == {"v": 2}


class _StalledLocalFileStorage(LocalFileStorage):
    """A filesystem that stalls: ``store`` is an ``async def`` whose body blocks
    without ever suspending — exactly what LocalFileStorage.store is, only slower."""

    stall_s = 0.2

    async def store(self, *args, **kwargs):
        time.sleep(self.stall_s)
        return await super().store(*args, **kwargs)


@pytest.mark.asyncio
async def test_a_stalled_store_does_not_pin_the_event_loop(tmp_path):
    """The write runs on a worker thread, so a stalled filesystem freezes neither
    the flusher's loop nor the coroutines sharing it (a concurrent recall, the
    pipeline awaiting drain())."""
    sink = StorageSink(StorageManager(_StalledLocalFileStorage(str(tmp_path))))
    run_id = str(uuid4())
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    try:
        await sink([_record(KIND_SUMMARY_GENERATED, run_id, None, "written")])
    finally:
        task.cancel()

    # ~20 ticks fit in a 0.2 s stall; on-loop the ticker would have seen 0-1.
    assert ticks >= 5
    [blob] = (tmp_path / "nodataset" / run_id / KIND_SUMMARY_GENERATED).glob("*.jsonl.gz")
    assert [line["payload"] for line in _read_jsonl_gz(blob)] == ["written"]


@pytest.mark.asyncio
async def test_sink_timeout_cuts_a_wedged_local_write(tmp_path):
    """SINK_TIMEOUT_S can only fire at a real suspension point. Awaited on the
    loop, LocalFileStorage.store had none, so a wedged write pinned the flusher
    past every timeout and drain()'s documented budget was false for the default
    sink. Off-loop, the timeout lands and the event is accounted for as dropped."""
    storage = _StalledLocalFileStorage(str(tmp_path))
    storage.stall_s = 1.0
    capture.register_capture_sink(StorageSink(StorageManager(storage)))
    hook._configure(flush_interval_s=60.0, sink_timeout_s=0.05)

    capture.emit(KIND_SUMMARY_GENERATED, "wedged", payload_kind="text")
    started = time.monotonic()
    await capture.drain(timeout=0.5)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5  # the 0.05 s timeout, not the 1 s stall
    assert hook._runtime.dropped == 1
    assert not hook._runtime.buffer
    assert hook._in_flight_total() == 0


@pytest.mark.asyncio
async def test_non_json_values_are_stringified(tmp_path):
    sink = StorageSink(StorageManager(LocalFileStorage(str(tmp_path))))
    run_id = uuid4()

    await sink([_record(KIND_SUMMARY_GENERATED, str(run_id), None, {"id": run_id})])

    [blob] = (tmp_path / "nodataset" / str(run_id) / KIND_SUMMARY_GENERATED).glob("*.jsonl.gz")
    assert _read_jsonl_gz(blob)[0]["payload"] == {"id": str(run_id)}
