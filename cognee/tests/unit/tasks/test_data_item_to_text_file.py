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
from unittest.mock import patch

import pytest

import cognee.tasks.ingestion.data_item_to_text_file as ditf
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
