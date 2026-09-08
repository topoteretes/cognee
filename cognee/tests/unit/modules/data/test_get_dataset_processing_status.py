"""Counting semantics of get_dataset_processing_status (SDK-23).

Rows are stand-ins with only the ``pipeline_status`` attribute the helper
reads; ``get_dataset_data`` is patched on the helper's module so no database
is involved.
"""

import importlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.data.methods import get_dataset_processing_status

# ``import a.b.c as module`` would bind the *function* here: the package
# re-exports it under the same name as the submodule. Resolve the module itself.
module = importlib.import_module("cognee.modules.data.methods.get_dataset_processing_status")

COMPLETED = "DATA_ITEM_PROCESSING_COMPLETED"


def _row(pipeline_status):
    return SimpleNamespace(pipeline_status=pipeline_status)


def _completed_row(dataset_id, pipeline="cognify_pipeline", value=COMPLETED):
    return _row({pipeline: {str(dataset_id): value}})


def _patch_rows(monkeypatch, rows):
    monkeypatch.setattr(module, "get_dataset_data", AsyncMock(return_value=rows))


@pytest.mark.asyncio
async def test_empty_dataset(monkeypatch):
    dataset_id = uuid.uuid4()
    _patch_rows(monkeypatch, [])

    assert await get_dataset_processing_status(dataset_id) == {
        "total": 0,
        "completed": 0,
        "pending": 0,
    }


@pytest.mark.asyncio
async def test_fully_completed_dataset(monkeypatch):
    dataset_id = uuid.uuid4()
    _patch_rows(monkeypatch, [_completed_row(dataset_id) for _ in range(3)])

    assert await get_dataset_processing_status(dataset_id) == {
        "total": 3,
        "completed": 3,
        "pending": 0,
    }


@pytest.mark.asyncio
async def test_mixed_statuses(monkeypatch):
    dataset_id = uuid.uuid4()
    _patch_rows(
        monkeypatch,
        [
            _completed_row(dataset_id),
            _row({}),  # freshly added, never cognified
            _row(None),  # legacy row with NULL pipeline_status
            _row({"cognify_pipeline": {}}),  # pipeline key present, no dataset slot
            _completed_row(dataset_id),
        ],
    )

    assert await get_dataset_processing_status(dataset_id) == {
        "total": 5,
        "completed": 2,
        "pending": 3,
    }


@pytest.mark.asyncio
async def test_legacy_string_and_dict_format_both_count(monkeypatch):
    """Backward compat: the bare string and the planned dict shape are both completed."""
    dataset_id = uuid.uuid4()
    _patch_rows(
        monkeypatch,
        [
            _completed_row(dataset_id, value=COMPLETED),
            _completed_row(dataset_id, value={"status": COMPLETED, "content_hash": "deadbeef"}),
            _completed_row(dataset_id, value={"status": "DATA_ITEM_PROCESSING_STARTED"}),
            _completed_row(dataset_id, value={"content_hash": "deadbeef"}),
        ],
    )

    assert await get_dataset_processing_status(dataset_id) == {
        "total": 4,
        "completed": 2,
        "pending": 2,
    }


@pytest.mark.asyncio
async def test_stamp_for_other_dataset_or_pipeline_is_pending(monkeypatch):
    dataset_id = uuid.uuid4()
    other_dataset_id = uuid.uuid4()
    _patch_rows(
        monkeypatch,
        [
            _completed_row(other_dataset_id),
            _completed_row(dataset_id, pipeline="add_pipeline"),
        ],
    )

    assert await get_dataset_processing_status(dataset_id) == {
        "total": 2,
        "completed": 0,
        "pending": 2,
    }


@pytest.mark.asyncio
async def test_pipeline_name_override(monkeypatch):
    dataset_id = uuid.uuid4()
    _patch_rows(
        monkeypatch,
        [
            _completed_row(dataset_id, pipeline="add_pipeline"),
            _completed_row(dataset_id, pipeline="cognify_pipeline"),
        ],
    )

    result = await get_dataset_processing_status(dataset_id, pipeline_name="add_pipeline")

    assert result == {"total": 2, "completed": 1, "pending": 1}
    module.get_dataset_data.assert_awaited_once_with(dataset_id)
