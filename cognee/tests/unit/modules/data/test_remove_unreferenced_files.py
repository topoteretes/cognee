"""Releasing stored files: every file is tried, and every one that stays is reported."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import cognee.modules.data.methods.remove_unreferenced_files as removal
from cognee.modules.data.exceptions import StoredFileNotRemovedError

OUTCOME = "The update was applied: the document holds its new content."


def _engine(monkeypatch, referenced=(), failing=(), outside=()):
    """A relational engine that records what it was asked to check and remove."""
    removed: list[str] = []

    async def remove_data_file(location):
        if location in failing:
            raise OSError(f"cannot delete {location}")
        removed.append(location)

    engine = SimpleNamespace(
        is_managed_data_file=lambda location: location not in outside,
        referenced_data_files=AsyncMock(return_value=set(referenced)),
        remove_data_file=remove_data_file,
        removed=removed,
    )
    monkeypatch.setattr(removal, "get_relational_engine", lambda: engine)
    return engine


@pytest.mark.asyncio
async def test_references_are_read_once_and_only_unused_files_are_removed(monkeypatch):
    engine = _engine(monkeypatch, referenced={"/data/b"}, outside={"/home/user/own.pdf"})

    await removal.remove_unreferenced_files(
        ["/data/a", "/data/b", None, "/data/a", "/home/user/own.pdf", "/data/c"], OUTCOME
    )

    # One query for every distinct file under the data root; the user's own file is not asked about.
    engine.referenced_data_files.assert_awaited_once_with(["/data/a", "/data/b", "/data/c"])
    assert engine.removed == ["/data/a", "/data/c"]


@pytest.mark.asyncio
async def test_every_file_is_tried_and_every_failure_is_named(monkeypatch):
    engine = _engine(monkeypatch, failing={"/data/a", "/data/c"})

    with pytest.raises(StoredFileNotRemovedError) as raised:
        await removal.remove_unreferenced_files(["/data/a", "/data/b", "/data/c"], OUTCOME)

    error = raised.value
    # The failure on the first file did not stop the others from being tried.
    assert engine.removed == ["/data/b"]
    assert error.locations == ["/data/a", "/data/c"]
    assert error.not_removed == {
        "/data/a": "OSError: cannot delete /data/a",
        "/data/c": "OSError: cannot delete /data/c",
    }
    assert error.outcome == OUTCOME
    message = str(error)
    assert OUTCOME in message and "failed for 2 file(s)" in message
    assert "/data/a (OSError: cannot delete /data/a)" in message
    assert "/data/c (OSError: cannot delete /data/c)" in message
    assert "/data/b" not in message
    assert isinstance(error.__cause__, OSError)


@pytest.mark.asyncio
async def test_removals_run_at_the_same_time(monkeypatch):
    engine = _engine(monkeypatch)
    in_flight = 0
    most_in_flight = 0

    async def remove_data_file(location):
        nonlocal in_flight, most_in_flight
        in_flight += 1
        most_in_flight = max(most_in_flight, in_flight)
        await asyncio.sleep(0)  # Where a storage round trip would wait.
        in_flight -= 1

    engine.remove_data_file = remove_data_file

    await removal.remove_unreferenced_files(["/data/a", "/data/b", "/data/c"], OUTCOME)

    # One after another this would be 1: a file's removal does not wait for the one before it.
    assert most_in_flight == 3


@pytest.mark.asyncio
async def test_a_reference_check_that_fails_reports_every_file(monkeypatch):
    engine = _engine(monkeypatch)
    engine.referenced_data_files = AsyncMock(side_effect=RuntimeError("database is down"))

    with pytest.raises(StoredFileNotRemovedError) as raised:
        await removal.remove_unreferenced_files(["/data/a", "/data/b"], OUTCOME)

    # Nothing could be checked, so nothing was removed, and both files are named.
    assert engine.removed == []
    assert raised.value.locations == ["/data/a", "/data/b"]
    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_nothing_under_the_data_root_asks_nothing(monkeypatch):
    engine = _engine(monkeypatch, outside={"/home/user/own.pdf"})

    await removal.remove_unreferenced_files([None, "/home/user/own.pdf"], OUTCOME)

    engine.referenced_data_files.assert_not_awaited()
    assert engine.removed == []
