import os
import tempfile

import pytest

from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage


@pytest.mark.asyncio
async def test_get_size_returns_zero_for_missing_file():
    """Regression test for the get_size() missing-await bug.

    file_exists() is async, so calling it without await left the coroutine
    (always truthy), made the ``else 0`` branch unreachable, and raised
    FileNotFoundError for a missing file instead of returning 0.
    """
    with tempfile.TemporaryDirectory() as tmp:
        storage = LocalFileStorage(tmp)
        assert await storage.get_size("does_not_exist.txt") == 0


@pytest.mark.asyncio
async def test_get_size_returns_actual_size_for_existing_file():
    with tempfile.TemporaryDirectory() as tmp:
        storage = LocalFileStorage(tmp)
        with open(os.path.join(tmp, "file.txt"), "wb") as handle:
            handle.write(b"hello")
        assert await storage.get_size("file.txt") == 5
