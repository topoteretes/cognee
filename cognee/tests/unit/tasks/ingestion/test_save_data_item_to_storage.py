import os
from unittest.mock import AsyncMock, patch

import pytest

from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage


@pytest.mark.asyncio
async def test_existing_absolute_path_returns_file_uri(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = await save_data_item_to_storage(str(file_path))

    assert result == file_path.as_uri()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="POSIX absolute-path semantics")
async def test_posix_absolute_path_behavior_unchanged_on_posix():
    # On POSIX, "/"-prefixed strings are genuine absolute paths and convert to
    # file URIs whether or not the file exists (pre-existing behavior).
    result = await save_data_item_to_storage("/nonexistent/path/file.txt")

    assert result == "file:///nonexistent/path/file.txt"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows drive-relative path semantics")
async def test_posix_style_string_falls_back_to_text_on_windows():
    """A "/"-prefixed string that is not a usable path on Windows is ingested as text.

    Regression test: os.path.normpath("/x") produces a drive-relative
    WindowsPath, and Path.as_uri() raised ValueError ("relative paths can't
    be expressed as file URIs") out of add() for any string starting with
    "/" — both POSIX-style paths and plain text.
    """
    with patch(
        "cognee.tasks.ingestion.save_data_item_to_storage.save_data_to_file",
        new_callable=AsyncMock,
        return_value="text-file-path",
    ) as mock_save:
        result = await save_data_item_to_storage("/remember to call Bob about the meeting")

    assert result == "text-file-path"
    mock_save.assert_awaited_once_with("/remember to call Bob about the meeting")


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows drive-relative path semantics")
async def test_drive_relative_string_falls_back_to_text_on_windows():
    # "C:name.txt" matches the Windows-path arm but is drive-relative, not
    # absolute; it used to raise the same ValueError from Path.as_uri().
    with patch(
        "cognee.tasks.ingestion.save_data_item_to_storage.save_data_to_file",
        new_callable=AsyncMock,
        return_value="text-file-path",
    ) as mock_save:
        result = await save_data_item_to_storage("C:drive-relative-note.txt")

    assert result == "text-file-path"
    mock_save.assert_awaited_once()
