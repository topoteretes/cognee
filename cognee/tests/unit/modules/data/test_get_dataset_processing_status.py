"""Counting and per-item semantics of get_dataset_processing_status (SDK-23).

Rows are stand-ins with only the attributes the helper reads; ``get_dataset_data``
is patched on the helper's module so no database is involved.
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


def _row(pipeline_status, name="doc.txt"):
    return SimpleNamespace(id=uuid.uuid4(), name=name, pipeline_status=pipeline_status)


def _completed_row(dataset_id, pipeline="cognify_pipeline", value=COMPLETED, name="doc.txt"):
    return _row({pipeline: {str(dataset_id): value}}, name=name)


def _patch_rows(monkeypatch, rows):
    monkeypatch.setattr(module, "get_dataset_data", AsyncMock(return_value=rows))


def _counts(result):
    return {key: result[key] for key in ("total", "completed", "pending")}


def _completed_flags(result):
    return [item["completed"] for item in result["items"]]


@pytest.mark.asyncio
async def test_empty_dataset(monkeypatch):
    _patch_rows(monkeypatch, [])

    result = await get_dataset_processing_status(uuid.uuid4())

    assert result == {"total": 0, "completed": 0, "pending": 0, "items": []}


@pytest.mark.asyncio
async def test_fully_completed_dataset(monkeypatch):
    dataset_id = uuid.uuid4()
    _patch_rows(monkeypatch, [_completed_row(dataset_id) for _ in range(3)])

    result = await get_dataset_processing_status(dataset_id)

    assert _counts(result) == {"total": 3, "completed": 3, "pending": 0}
    assert _completed_flags(result) == [True, True, True]


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

    result = await get_dataset_processing_status(dataset_id)

    assert _counts(result) == {"total": 5, "completed": 2, "pending": 3}
    assert _completed_flags(result) == [True, False, False, False, True]


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

    result = await get_dataset_processing_status(dataset_id)

    assert _counts(result) == {"total": 4, "completed": 2, "pending": 2}
    assert _completed_flags(result) == [True, True, False, False]


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

    result = await get_dataset_processing_status(dataset_id)

    assert _counts(result) == {"total": 2, "completed": 0, "pending": 2}
    assert _completed_flags(result) == [False, False]


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

    assert _counts(result) == {"total": 2, "completed": 1, "pending": 1}
    assert _completed_flags(result) == [True, False]
    module.get_dataset_data.assert_awaited_once_with(dataset_id)


@pytest.mark.asyncio
async def test_items_carry_data_id_and_name_in_storage_order(monkeypatch):
    dataset_id = uuid.uuid4()
    done = _completed_row(dataset_id, name="done.pdf")
    fresh = _row({}, name="fresh.md")
    legacy_null = _row(None, name="legacy.txt")
    _patch_rows(monkeypatch, [done, fresh, legacy_null])

    result = await get_dataset_processing_status(dataset_id)

    assert result == {
        "total": 3,
        "completed": 1,
        "pending": 2,
        "items": [
            {"id": done.id, "name": "done.pdf", "completed": True},
            {"id": fresh.id, "name": "fresh.md", "completed": False},
            {"id": legacy_null.id, "name": "legacy.txt", "completed": False},
        ],
    }
