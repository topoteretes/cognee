"""A pinned data_id resurfacing with different content must fail loudly.

The incremental pre-check used to short-circuit on "pipeline already
completed" before any content comparison, silently keeping the stale record
(observed live: a changed code-repo manifest kept its old file_count). Now:
same content -> the cheap idempotent skip stays; different content ->
DataContentConflictError naming the ways out; non-pinned items are untouched
(their identity derives from content, so they cannot conflict).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.run_tasks_data_item as rtdi
from cognee.modules.ingestion.exceptions import DataContentConflictError
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunAlreadyCompleted
from cognee.tasks.ingestion.data_item import DataItem

PIPELINE = "add_pipeline"


def _dataset():
    return SimpleNamespace(id=uuid4(), name="conflict_ds", owner_id=uuid4())


def _completed_record(dataset, content_hash="hash-v1"):
    return SimpleNamespace(
        pipeline_status={
            PIPELINE: {str(dataset.id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}
        },
        content_hash=content_hash,
    )


def _fake_engine(data_point):
    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = data_point
    session.execute = AsyncMock(return_value=result)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock()
    engine.get_async_session.return_value = ctx
    return engine


def _run(data_item, dataset, engine, incoming_hash):
    return rtdi.run_tasks_data_item_incremental(
        data_item=data_item,
        dataset=dataset,
        tasks=[],
        pipeline_name=PIPELINE,
        pipeline_id="pid",
        pipeline_run_id=uuid4(),
        ctx=None,
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
    )


@pytest.mark.asyncio
async def test_same_content_keeps_the_idempotent_skip():
    dataset = _dataset()
    item = DataItem(data="manifest v1", data_id=uuid4())
    with (
        patch.object(
            rtdi, "get_relational_engine", return_value=_fake_engine(_completed_record(dataset))
        ),
        patch.object(rtdi, "_pinned_item_content_hash", new=AsyncMock(return_value="hash-v1")),
    ):
        results = [step async for step in _run(item, dataset, None, "hash-v1")]

    assert len(results) == 1
    assert isinstance(results[0]["run_info"], PipelineRunAlreadyCompleted)


@pytest.mark.asyncio
async def test_changed_content_raises_conflict_with_remedies():
    dataset = _dataset()
    item = DataItem(data="manifest v2 CHANGED", data_id=uuid4())
    with (
        patch.object(
            rtdi, "get_relational_engine", return_value=_fake_engine(_completed_record(dataset))
        ),
        patch.object(rtdi, "_pinned_item_content_hash", new=AsyncMock(return_value="hash-v2")),
    ):
        with pytest.raises(DataContentConflictError) as excinfo:
            async for _step in _run(item, dataset, None, "hash-v2"):
                pass

    message = str(excinfo.value)
    assert "update()" in message
    assert "different data_id" in message
    assert "incremental_loading=False and data_cache=False" in message


@pytest.mark.asyncio
async def test_undeterminable_hash_keeps_legacy_skip():
    """Items whose incoming hash cannot be computed (e.g. already-persisted
    Data instances) keep the pre-existing silent skip - no false conflicts."""
    dataset = _dataset()
    item = DataItem(data="anything", data_id=uuid4())
    with (
        patch.object(
            rtdi, "get_relational_engine", return_value=_fake_engine(_completed_record(dataset))
        ),
        patch.object(rtdi, "_pinned_item_content_hash", new=AsyncMock(return_value=None)),
    ):
        results = [step async for step in _run(item, dataset, None, None)]

    assert isinstance(results[0]["run_info"], PipelineRunAlreadyCompleted)


@pytest.mark.asyncio
async def test_hash_helper_returns_none_for_unpinned_items():
    assert await rtdi._pinned_item_content_hash("plain text") is None
    assert await rtdi._pinned_item_content_hash(DataItem(data="text", data_id=None)) is None
