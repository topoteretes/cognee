"""Tests for run_tasks_data_item_incremental.

Covers the fix where the saved file path (not the consumed UploadFile stream)
is forwarded to pipeline tasks after save_data_item_to_storage.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.pipelines.operations.run_tasks_data_item import (
    run_tasks_data_item_incremental,
)
from cognee.tasks.ingestion.data_item import DataItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dataset():
    ds = MagicMock()
    ds.id = uuid4()
    ds.name = "test_dataset"
    return ds


@pytest.fixture()
def user():
    u = MagicMock()
    u.id = uuid4()
    u.tenant_id = None
    return u


@asynccontextmanager
async def _fake_open(path, mode="rb", **kw):
    yield MagicMock()


def _patch_incremental(saved_path, data_id):
    """Return a combined context manager that patches all dependencies."""
    db_data_point = MagicMock()
    db_data_point.pipeline_status = {}

    first_session = AsyncMock()
    first_session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )

    update_session = AsyncMock()
    update_session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=db_data_point))
    )
    update_session.merge = AsyncMock()
    update_session.commit = AsyncMock()

    call_count = 0

    @asynccontextmanager
    async def fake_session():
        nonlocal call_count
        call_count += 1
        yield first_session if call_count == 1 else update_session

    engine = MagicMock()
    engine.get_async_session = fake_session

    captured = {}

    async def capture_run(**kwargs):
        captured.update(kwargs)
        yield {"ok": True}

    class _Ctx:
        """Holds captured kwargs after the run."""

        data = captured

    ctx = _Ctx()

    from contextlib import ExitStack

    stack = ExitStack()
    patches = [
        patch(
            "cognee.modules.pipelines.operations.run_tasks_data_item.save_data_item_to_storage",
            new=AsyncMock(return_value=saved_path),
        ),
        patch(
            "cognee.modules.pipelines.operations.run_tasks_data_item.open_data_file",
            new=_fake_open,
        ),
        patch(
            "cognee.modules.pipelines.operations.run_tasks_data_item.ingestion.classify",
            new=lambda f: MagicMock(),
        ),
        patch(
            "cognee.modules.pipelines.operations.run_tasks_data_item.ingestion.identify",
            new=AsyncMock(return_value=data_id),
        ),
        patch(
            "cognee.modules.pipelines.operations.run_tasks_data_item.get_relational_engine",
            return_value=engine,
        ),
        patch(
            "cognee.modules.pipelines.operations.run_tasks_data_item.run_tasks_with_telemetry",
            new=capture_run,
        ),
    ]
    for p in patches:
        stack.enter_context(p)

    return stack, ctx


async def _run(data_item, dataset, user):
    """Drain the async generator and return nothing (side effects only)."""
    async for _ in run_tasks_data_item_incremental(
        data_item=data_item,
        dataset=dataset,
        tasks=[],
        pipeline_name="test_pipeline",
        pipeline_id="test_pipeline_id",
        pipeline_run_id=str(uuid4()),
        context=None,
        user=user,
    ):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_file_passes_saved_path(dataset, user):
    """UploadFile-like objects should be replaced with the saved file path."""
    upload = MagicMock()
    upload.file = MagicMock()
    upload.filename = "report.pdf"
    saved = "/data/storage/report.pdf"

    stack, ctx = _patch_incremental(saved, uuid4())
    with stack:
        await _run(upload, dataset, user)

    assert ctx.data["data"] == [saved]


@pytest.mark.asyncio
async def test_data_instance_passed_through(dataset, user):
    """Data model instances must be forwarded as-is, without saving."""
    from cognee.modules.data.models import Data

    data_item = MagicMock(spec=Data)
    data_item.id = uuid4()
    data_item.pipeline_status = {}

    mock_save = AsyncMock(return_value="/unused")
    stack, ctx = _patch_incremental("/unused", uuid4())

    # Override save mock to track calls
    with (
        stack,
        patch(
            "cognee.modules.pipelines.operations.run_tasks_data_item.save_data_item_to_storage",
            new=mock_save,
        ),
    ):
        await _run(data_item, dataset, user)

    assert ctx.data["data"] == [data_item]
    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_dataitem_label_preserved(dataset, user):
    """DataItem labels should survive the file-path substitution."""
    item = DataItem(data="some text", label="my_label")
    saved = "/data/storage/text.txt"

    stack, ctx = _patch_incremental(saved, uuid4())
    with stack:
        await _run(item, dataset, user)

    result = ctx.data["data"][0]
    assert isinstance(result, DataItem)
    assert result.data == saved
    assert result.label == "my_label"


@pytest.mark.asyncio
async def test_dataitem_without_label_passes_plain_path(dataset, user):
    """DataItem with label=None should yield a plain file path string."""
    item = DataItem(data="content", label=None)
    saved = "/data/storage/content.txt"

    stack, ctx = _patch_incremental(saved, uuid4())
    with stack:
        await _run(item, dataset, user)

    assert ctx.data["data"] == [saved]
