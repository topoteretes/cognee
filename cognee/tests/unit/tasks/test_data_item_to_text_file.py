"""Regression test for the Windows temp-file reopen bug in
``data_item_to_text_file`` (S3 ingestion path).

``tempfile.NamedTemporaryFile`` defaults to ``delete=True``, which on Windows
keeps an exclusive lock on the file and forbids reopening it *by name* while the
handle is still open. The S3 branch writes the downloaded bytes to such a temp
file and then hands ``temp_file.name`` to the loader, which reopens it by name --
raising ``PermissionError`` on Windows.

The fix creates the temp file with ``delete=False``, closes the handle before the
loader reopens it, and unlinks it afterwards. These tests lock that in:

* the loader can reopen the file by name and read the written bytes (strict on
  Windows, benign on POSIX), and
* the temp file is created with ``delete=False`` and removed afterwards
  (cross-platform guard -- a revert to ``delete=True`` fails on every OS).
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import cognee.tasks.ingestion.data_item_to_text_file as ditf
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.tasks.ingestion.data_item_to_text_file import data_item_to_text_file


class _FakeLoader:
    loader_name = "fake_loader"


class _FakeLoaderEngine:
    """Stand-in for the loader engine that reopens the temp file *by name*,
    mirroring what the real ``load_file`` / ``get_loader`` do."""

    def __init__(self):
        self.content_seen = None

    async def load_file(self, file_path, preferred_loaders=None, **kwargs):
        # The real loader reopens the path by name; on Windows this is exactly
        # where the buggy delete=True handle raises PermissionError.
        with open(file_path, "rb") as f:
            self.content_seen = f.read()
        return "STORAGE_PATH"

    def get_loader(self, file_path, preferred_loaders=None):
        # The real get_loader also touches the file (guess_file_type).
        with open(file_path, "rb"):
            pass
        return _FakeLoader()


@pytest.mark.asyncio
async def test_s3_temp_file_reopened_by_name_and_cleaned_up():
    engine = _FakeLoaderEngine()
    captured = {}

    async def fake_pull_from_s3(file_path, destination_file):
        destination_file.write(b"payload-bytes")

    real_named_temp_file = tempfile.NamedTemporaryFile

    def spy_named_temp_file(*args, **kwargs):
        captured["kwargs"] = kwargs
        handle = real_named_temp_file(*args, **kwargs)
        captured["name"] = handle.name
        return handle

    with (
        patch.object(ditf, "pull_from_s3", new=fake_pull_from_s3),
        patch.object(ditf, "get_loader_engine", return_value=engine),
        patch.object(ditf.tempfile, "NamedTemporaryFile", side_effect=spy_named_temp_file),
    ):
        storage_path, loader = await data_item_to_text_file("s3://bucket/key.txt")

    # Functional contract preserved.
    assert storage_path == "STORAGE_PATH"
    assert isinstance(loader, _FakeLoader)

    # Reopen-by-name succeeded (strict on Windows, benign on POSIX).
    assert engine.content_seen == b"payload-bytes"

    # Cross-platform regression guard: the temp file must be created with
    # delete=False so the handle can be closed before the loader reopens it.
    assert captured["kwargs"].get("delete") is False

    # And it must still be cleaned up afterwards (no temp-file leak).
    assert not os.path.exists(captured["name"])


@pytest.mark.asyncio
async def test_internal_storage_accepted_when_accept_local_file_path_false(monkeypatch):
    """Internal storage files must be accepted by data_item_to_text_file even when
    accept_local_file_path is False (e.g. multipart uploads saved to cognee storage)."""
    monkeypatch.setattr(ditf.settings, "accept_local_file_path", False)
    mock_loader = _FakeLoader()
    mock_engine = MagicMock()
    mock_engine.load_file = AsyncMock(return_value="OUTPUT_PATH")
    mock_engine.get_loader = MagicMock(return_value=mock_loader)
    monkeypatch.setattr(ditf, "get_loader_engine", lambda: mock_engine)

    # Absolute path with is_internal_storage=True
    abs_path = "C:\\cognee\\storage\\test.txt" if os.name == "nt" else "/cognee/storage/test.txt"
    storage_path, loader = await data_item_to_text_file(abs_path, is_internal_storage=True)
    assert storage_path == "OUTPUT_PATH"
    assert loader is mock_loader

    # file:// URI with is_internal_storage=True
    file_uri = "file:///cognee/storage/test.txt"
    storage_path, loader = await data_item_to_text_file(file_uri, is_internal_storage=True)
    assert storage_path == "OUTPUT_PATH"
    assert loader is mock_loader


@pytest.mark.asyncio
async def test_external_local_file_rejected_when_accept_local_file_path_false(monkeypatch):
    """External local files must be rejected when accept_local_file_path is False
    and is_internal_storage is False."""
    monkeypatch.setattr(ditf.settings, "accept_local_file_path", False)

    # Absolute path with default / False is_internal_storage
    abs_path = "C:\\external\\note.txt" if os.name == "nt" else "/external/note.txt"
    with pytest.raises(IngestionError, match="Local files are not accepted"):
        await data_item_to_text_file(abs_path)

    with pytest.raises(IngestionError, match="Local files are not accepted"):
        await data_item_to_text_file(abs_path, is_internal_storage=False)

    # file:// URI with default / False is_internal_storage
    file_uri = "file:///external/note.txt"
    with pytest.raises(IngestionError, match="Local files are not accepted"):
        await data_item_to_text_file(file_uri)

    with pytest.raises(IngestionError, match="Local files are not accepted"):
        await data_item_to_text_file(file_uri, is_internal_storage=False)
