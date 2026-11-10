import importlib
import os
from unittest.mock import AsyncMock

import pytest

from cognee.modules.ingestion import StoredFile
from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage

# The package __init__ rebinds the name ``save_data_item_to_storage`` to the
# function, so ``import ... as mod`` would yield the function, not the module.
# import_module returns the real module object (for patching its globals).
mod = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")


@pytest.mark.asyncio
async def test_existing_absolute_file_returns_file_uri(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = await save_data_item_to_storage(str(file_path))

    assert result == file_path.as_uri()


@pytest.mark.asyncio
async def test_nonexistent_absolute_path_is_ingested_as_text(monkeypatch):
    """A "/"-prefixed string that is not an existing file is ingested as text.

    Regression test for #3887. On every platform, a plain text note that happens
    to start with "/" (e.g. slash-command-style content) must be saved as text
    rather than turned into a file:// URI for a file that does not exist. On
    Windows the pre-fix code additionally crashed here with
    ``ValueError: relative path can't be expressed as a file URI``.
    """
    save_mock = AsyncMock(return_value=StoredFile(file_path="text-file-path"))
    monkeypatch.setattr(mod, "save_data_to_file_detailed", save_mock)

    note = "/remember to call Bob about the meeting"
    result = await save_data_item_to_storage(note)

    assert result == "text-file-path"
    save_mock.assert_awaited_once_with(note)


@pytest.mark.asyncio
async def test_windows_style_paths_do_not_crash_and_fall_back_to_text(monkeypatch):
    """A "/"-prefixed or drive-relative string that names no existing file is
    text, never a broken file:// URI (#3887's ``ValueError: relative path can't
    be expressed as a file URI``). The lookup resolves the string to an absolute
    path before it builds a URI, and only for an existing file. The genuine
    drive-anchored Windows path (``C:\\...``) that points at an existing file is
    covered by the Windows-only test below and the OS-matrix CI.
    """
    save_mock = AsyncMock(return_value=StoredFile(file_path="text-file-path"))
    monkeypatch.setattr(mod, "save_data_to_file_detailed", save_mock)

    for item in ["/no/such/windows/path.txt", "C:name.txt"]:
        save_mock.reset_mock()
        result = await save_data_item_to_storage(item)
        assert result == "text-file-path"
        save_mock.assert_awaited_once_with(item)


@pytest.mark.asyncio
async def test_absolute_path_is_text_whether_or_not_it_exists_when_gate_disabled(
    tmp_path, monkeypatch
):
    # With ACCEPT_LOCAL_FILE_PATH off the disk is never asked, so an existing
    # file and a missing one get the same answer: text. That is what keeps a
    # reply from revealing whether a path exists; explicit paths are refused
    # earlier, by screen_server_paths at the entry.
    existing = tmp_path / "note.txt"
    existing.write_text("hello", encoding="utf-8")
    save_mock = AsyncMock(return_value=StoredFile(file_path="text-file-path"))
    monkeypatch.setattr(mod, "save_data_to_file_detailed", save_mock)
    monkeypatch.setattr(mod.settings, "accept_local_file_path", False)

    for item in (str(existing), "/no/such/file.txt"):
        save_mock.reset_mock()
        assert await save_data_item_to_storage(item) == "text-file-path"
        save_mock.assert_awaited_once_with(item)


@pytest.mark.asyncio
async def test_existing_directory_is_ingested_as_text(tmp_path, monkeypatch):
    # An existing directory is not a file (is_file() is False), so it falls
    # through to text ingestion rather than becoming a file:// URI. (In the
    # normal add() flow directories are expanded upstream by
    # resolve_data_directories.) The dry-run estimator deliberately diverges
    # here — it rejects directories with a clearer message; see
    # test_estimator.test_directories_are_rejected.
    save_mock = AsyncMock(return_value=StoredFile(file_path="text-file-path"))
    monkeypatch.setattr(mod, "save_data_to_file_detailed", save_mock)

    result = await save_data_item_to_storage(str(tmp_path))

    assert result == "text-file-path"
    save_mock.assert_awaited_once_with(str(tmp_path))


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows absolute-path semantics")
async def test_genuine_windows_absolute_path_returns_file_uri(tmp_path):
    # A real Windows absolute path (C:\...) to an existing file still converts
    # to a file:// URI. Runs only on Windows, where tmp_path is drive-anchored.
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = await save_data_item_to_storage(str(file_path))

    assert result == file_path.as_uri()
