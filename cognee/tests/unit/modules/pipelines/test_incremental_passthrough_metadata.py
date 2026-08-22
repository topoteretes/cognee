"""Pass-through items are hashed while their file is open.

For items cognee did not write (a local path, an ``s3://`` URL),
``run_tasks_data_item_incremental`` has to read the file to resolve the dedup
identity. The read happens through the ``IngestionData`` metadata accessors —
so it MUST run inside the ``open_data_file`` block. A regression here slips
past mocked-I/O tests (the file object never really closes) and past upload
benchmarks (uploads carry metadata and skip this branch entirely), so this test
uses the real storage stack over a real temp file.
"""

import hashlib
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import cognee.modules.pipelines.operations.run_tasks_data_item as item_module
from cognee.modules.ingestion import StoredFile
from cognee.tasks.ingestion.carried_source import CARRIED_SOURCE_KEY
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus

CONTENT = b"pass-through payload, hashed while open"


@pytest.mark.asyncio
async def test_passthrough_item_is_hashed_while_its_file_is_open(tmp_path, monkeypatch):
    source = tmp_path / "note.txt"
    source.write_bytes(CONTENT)

    # The item is a pass-through: storage hands the path back, no metadata.
    monkeypatch.setattr(
        item_module,
        "save_data_item_to_storage_detailed",
        AsyncMock(return_value=StoredFile(file_path=source.as_uri(), metadata=None)),
    )

    seen = {}

    async def capture_identify(content_hash, user, dataset_id):
        seen["content_hash"] = content_hash
        return None

    monkeypatch.setattr(item_module.ingestion, "identify_data_by_hash", capture_identify)
    # local_path_safety allowlists cognee's own roots; tmp_path is not one.
    monkeypatch.setattr(
        item_module,
        "open_data_file",
        __import__(
            "cognee.infrastructure.files.storage.LocalFileStorage", fromlist=["LocalFileStorage"]
        )
        .LocalFileStorage(str(tmp_path))
        .open,
    )
    monkeypatch.setattr(item_module, "get_relational_engine", lambda: SimpleNamespace())

    async def _no_tasks(**kwargs):
        return
        yield  # pragma: no cover

    monkeypatch.setattr(item_module, "run_tasks_with_telemetry", lambda **kw: _no_tasks(**kw))

    ctx = SimpleNamespace(extras={})
    dataset = SimpleNamespace(id=uuid4(), name="ds")

    outputs = []
    async for out in item_module.run_tasks_data_item_incremental(
        data_item=source.as_uri(),
        dataset=dataset,
        tasks=[],
        pipeline_name="p",
        pipeline_id="p",
        pipeline_run_id=uuid4(),
        ctx=ctx,
        user=SimpleNamespace(id=uuid4()),
    ):
        outputs.append(out)
        break  # identity resolution happens before the first yield

    # The dedup identity was really read from the file — a read attempted after
    # the ``open_data_file`` block closes raises instead of producing this hash.
    assert seen["content_hash"] == hashlib.md5(CONTENT).hexdigest()

    # And the work is handed to ingest_data, so it does not re-read the file.
    carried = ctx.extras[CARRIED_SOURCE_KEY]
    assert carried.stored.metadata["content_hash"] == hashlib.md5(CONTENT).hexdigest()
    assert carried.stored.metadata["file_size"] == len(CONTENT)
