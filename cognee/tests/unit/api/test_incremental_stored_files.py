"""An incremental update keeps no stored file that no ``Data`` row points at."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.modules.data.methods.remove_unreferenced_files as removal
from cognee.api.v1.update import incremental
from cognee.modules.data.exceptions import StoredFileNotRemovedError

OLD = SimpleNamespace(
    raw_data_location="/data/text_old.txt",
    original_data_location="/data/a.pdf",
    content_hash="old-hash",
)
STAGED = SimpleNamespace(
    raw_data_location="/data/text_new.txt",
    original_data_location="/data/staged_b.pdf",
    content_hash="new-hash",
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
    monkeypatch.setattr(incremental, "record_data_item_content", AsyncMock())
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
    """No row points at any of the files, and every removal fails."""
    engine = SimpleNamespace(
        is_managed_data_file=lambda location: True,
        referenced_data_files=AsyncMock(return_value=set()),
        remove_data_file=AsyncMock(side_effect=OSError("storage is down")),
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
    engine = SimpleNamespace(
        is_managed_data_file=lambda location: True,
        referenced_data_files=AsyncMock(return_value={"/data/text_new.txt", "/data/staged_b.pdf"}),
        remove_data_file=AsyncMock(),
    )
    monkeypatch.setattr(removal, "get_relational_engine", lambda: engine)
    # Raw text is stored once: the row's processed text and its original are one file.
    text_row = SimpleNamespace(
        raw_data_location="/data/text_old.txt", original_data_location="/data/text_old.txt"
    )

    await incremental._remove_unreferenced_files(text_row, STAGED, update_applied=True)

    engine.referenced_data_files.assert_awaited_once_with(
        ["/data/text_old.txt", "/data/text_new.txt", "/data/staged_b.pdf"]
    )
    # The row now points at the staged files, so only the old one goes.
    engine.remove_data_file.assert_awaited_once_with("/data/text_old.txt")


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
    # Both files of the old version and both staged ones were tried, and all four are named.
    assert error.locations == [
        OLD.raw_data_location,
        OLD.original_data_location,
        STAGED.raw_data_location,
        STAGED.original_data_location,
    ]
    assert str(error).startswith(
        "StoredFileNotRemovedError: The update was applied: the document holds its new content."
    )
    assert "delete them from storage by hand" in str(error)
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


def _staging(monkeypatch, *, loader_result, open_error=None):
    """Staging stores ``/data/staged.txt``, then runs the loader; records what it releases."""
    released: list = []

    async def remove_files(locations, outcome):
        released.append((list(locations), outcome))

    monkeypatch.setattr(
        incremental, "save_data_item_to_storage", AsyncMock(return_value="/data/staged.txt")
    )
    monkeypatch.setattr(
        incremental, "data_item_to_text_file", AsyncMock(return_value=loader_result)
    )
    monkeypatch.setattr(incremental, "remove_unreferenced_files", remove_files)
    if open_error is not None:

        def failing_open(location):
            raise open_error

        monkeypatch.setattr(incremental, "open_data_file", failing_open)
    return released


@pytest.mark.asyncio
async def test_staging_no_loader_accepts_releases_the_original_it_stored(monkeypatch):
    released = _staging(monkeypatch, loader_result=(None, None))

    with pytest.raises(incremental.IncrementalUpdateNotPossible) as refused:
        await incremental._stage_new_content("new text", None)

    # The refusal still reaches update(), which goes on to the full rebuild under the
    # real filename: nothing would ever point at the staged original otherwise.
    assert refused.value.reason == incremental.RefusalReason.UNREADABLE_TEXT
    assert released == [(["/data/staged.txt"], incremental.UPDATE_NOT_APPLIED)]


@pytest.mark.asyncio
async def test_staging_that_fails_after_the_loader_releases_both_files(monkeypatch):
    loader = SimpleNamespace(loader_name="text_loader")
    released = _staging(
        monkeypatch,
        loader_result=("/data/text_new.txt", loader),
        open_error=OSError("cannot read"),
    )

    with pytest.raises(OSError, match="cannot read"):
        await incremental._stage_new_content("new text", None)

    assert released == [
        (["/data/staged.txt", "/data/text_new.txt"], incremental.UPDATE_NOT_APPLIED)
    ]
