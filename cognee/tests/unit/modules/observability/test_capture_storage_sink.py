"""StorageSink layout (SDK-529): gzip'd JSONL blobs per (dataset, run, kind) and a
pretty manifest.json per run, written through the normal StorageManager."""

import gzip
import json
import re
from uuid import uuid4

import pytest

from cognee.infrastructure.files.storage import StorageManager
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.modules.observability.capture import (
    KIND_RUN_MANIFEST,
    KIND_SUMMARY_GENERATED,
    StorageSink,
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
async def test_successive_writes_never_clobber_and_write_alias_exists(tmp_path):
    sink = StorageSink(StorageManager(LocalFileStorage(str(tmp_path))))
    run_id = str(uuid4())

    await sink([_record(KIND_SUMMARY_GENERATED, run_id, None, "a")])
    await sink.write([_record(KIND_SUMMARY_GENERATED, run_id, None, "b")])

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


@pytest.mark.asyncio
async def test_non_json_values_are_stringified(tmp_path):
    sink = StorageSink(StorageManager(LocalFileStorage(str(tmp_path))))
    run_id = uuid4()

    await sink([_record(KIND_SUMMARY_GENERATED, str(run_id), None, {"id": run_id})])

    [blob] = (tmp_path / "nodataset" / str(run_id) / KIND_SUMMARY_GENERATED).glob("*.jsonl.gz")
    assert _read_jsonl_gz(blob)[0]["payload"] == {"id": str(run_id)}
