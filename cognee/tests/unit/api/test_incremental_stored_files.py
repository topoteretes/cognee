"""An incremental update keeps no stored file that no ``Data`` row points at."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.modules.data.methods.remove_unreferenced_files as removal
from cognee.api.v1.update import incremental
from cognee.modules.data.exceptions import StoredFileNotRemovedError

OLD = SimpleNamespace(raw_data_location="/data/text_old.txt", original_data_location="/data/a.pdf")
STAGED = SimpleNamespace(
    raw_data_location="/data/text_new.txt", original_data_location="/data/staged_b.pdf"
)
_REAL_REMOVE_UNREFERENCED_FILES = incremental._remove_unreferenced_files


def _patch_run(monkeypatch, bundle, write):
    """Run ``_run_incremental_update`` over ``bundle`` and record the order of its steps."""
    order: list = []

    async def record_write(*args, **kwargs):
        order.append("write")
        return await write(*args, **kwargs)

    async def remove_files(*versions, update_applied):
        order.append(("remove files", versions, update_applied))

    async def complete(*args):
        order.append("run completed")

    monkeypatch.setattr(incremental, "_stage_and_plan", AsyncMock(return_value=bundle))
    monkeypatch.setattr(incremental, "_remove_unreferenced_files", remove_files)
    monkeypatch.setattr(incremental, "generate_pipeline_id", lambda *args: uuid4())
    monkeypatch.setattr(
        incremental,
        "log_pipeline_run_start",
        AsyncMock(return_value=SimpleNamespace(pipeline_run_id=uuid4())),
    )
    monkeypatch.setattr(incremental, "log_pipeline_run_complete", complete)
    monkeypatch.setattr(incremental, "log_pipeline_run_error", AsyncMock())
    monkeypatch.setattr(incremental, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(incremental, "_write_and_publish", record_write)
    monkeypatch.setattr(incremental, "retire_decisions_after_edit", AsyncMock(return_value=set()))
    return order


def _storage_down(monkeypatch):
    """Every removal fails, as when the storage backend cannot delete."""
    engine = SimpleNamespace(
        remove_data_file_if_unreferenced=AsyncMock(side_effect=OSError("storage is down"))
    )
    monkeypatch.setattr(removal, "get_relational_engine", lambda: engine)


async def _run():
    dataset, user = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
    return await incremental._run_incremental_update(
        uuid4(), "new text", dataset, user, None, None, None, None, None, None, None, "llm"
    )


@pytest.mark.asyncio
async def test_a_published_update_checks_the_old_and_the_staged_files_last(monkeypatch):
    bundle = {"status": "changed", "data_item": OLD, "staged": STAGED}
    order = _patch_run(monkeypatch, bundle, AsyncMock(return_value={}))

    await _run()

    assert order == ["write", "run completed", ("remove files", (OLD, STAGED), True)]


@pytest.mark.asyncio
async def test_a_failed_write_still_checks_the_files(monkeypatch):
    bundle = {"status": "changed", "data_item": OLD, "staged": STAGED}
    order = _patch_run(monkeypatch, bundle, AsyncMock(side_effect=RuntimeError("write failed")))

    with pytest.raises(RuntimeError, match="write failed"):
        await _run()

    assert order == ["write", ("remove files", (OLD, STAGED), False)]


@pytest.mark.asyncio
async def test_an_unchanged_update_with_nothing_to_repair_still_checks_the_files(monkeypatch):
    bundle = {
        "status": "unchanged",
        "repairs": [],
        "stored_count": 3,
        "data_item": OLD,
        "staged": STAGED,
    }
    order = _patch_run(monkeypatch, bundle, AsyncMock())

    result = await _run()

    assert result["status"] == "unchanged" and result["pipeline_run_id"] is None
    assert order == [("remove files", (OLD, STAGED), False)]


@pytest.mark.asyncio
async def test_each_distinct_location_is_offered_for_removal_once(monkeypatch):
    engine = SimpleNamespace(remove_data_file_if_unreferenced=AsyncMock())
    monkeypatch.setattr(removal, "get_relational_engine", lambda: engine)
    # Raw text is stored once: the row's processed text and its original are one file.
    text_row = SimpleNamespace(
        raw_data_location="/data/text_old.txt", original_data_location="/data/text_old.txt"
    )

    await incremental._remove_unreferenced_files(text_row, STAGED, update_applied=True)

    offered = [call.args[0] for call in engine.remove_data_file_if_unreferenced.await_args_list]
    assert offered == ["/data/text_old.txt", "/data/text_new.txt", "/data/staged_b.pdf"]


@pytest.mark.asyncio
async def test_a_file_that_cannot_be_removed_fails_an_update_that_was_published(monkeypatch):
    """Never reported as a success, and the error says the update is in effect."""
    bundle = {"status": "changed", "data_item": OLD, "staged": STAGED}
    order = _patch_run(monkeypatch, bundle, AsyncMock(return_value={}))
    monkeypatch.setattr(incremental, "_remove_unreferenced_files", _REAL_REMOVE_UNREFERENCED_FILES)
    _storage_down(monkeypatch)

    with pytest.raises(StoredFileNotRemovedError) as raised:
        await _run()

    assert order == ["write", "run completed"]
    error = raised.value
    assert error.outcome == incremental.UPDATE_APPLIED
    assert error.location == OLD.raw_data_location
    assert str(error).startswith(
        "StoredFileNotRemovedError: The update was applied: the document holds its new content."
    )
    assert "delete it from storage by hand" in str(error)
    assert isinstance(error.__cause__, OSError)


@pytest.mark.asyncio
async def test_a_file_that_cannot_be_removed_after_a_failed_write_says_nothing_was_applied(
    monkeypatch,
):
    bundle = {"status": "changed", "data_item": OLD, "staged": STAGED}
    _patch_run(monkeypatch, bundle, AsyncMock(side_effect=RuntimeError("write failed")))
    monkeypatch.setattr(incremental, "_remove_unreferenced_files", _REAL_REMOVE_UNREFERENCED_FILES)
    _storage_down(monkeypatch)

    with pytest.raises(StoredFileNotRemovedError) as raised:
        await _run()

    error = raised.value
    assert error.outcome == incremental.UPDATE_NOT_APPLIED
    assert "The update was not applied" in str(error)
    # The storage error is the cause, and the write error it interrupted is kept behind it.
    assert isinstance(error.__cause__, OSError)
    assert isinstance(error.__cause__.__context__, RuntimeError)
    assert str(error.__cause__.__context__) == "write failed"
