"""``S3FileStorage.list_files`` must not report a failed listing as an empty one.

Only a missing directory/prefix means "no files"; permission, transport and
endpoint errors propagate, matching ``LocalFileStorage.list_files``.
"""

import pytest

from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage


class _FakeS3:
    def __init__(self, glob_error=None, paths=()):
        self._glob_error = glob_error
        self._paths = list(paths)

    def glob(self, pattern):
        if self._glob_error is not None:
            raise self._glob_error
        return self._paths

    def isfile(self, path):
        return True


def _storage(s3):
    # Bypass __init__: it imports s3fs and reads the S3 config, neither of
    # which the listing logic needs.
    storage = S3FileStorage.__new__(S3FileStorage)
    storage.storage_path = "s3://bucket/root"
    storage.s3 = s3
    return storage


@pytest.mark.asyncio
async def test_missing_directory_lists_as_empty():
    storage = _storage(_FakeS3(glob_error=FileNotFoundError("no such prefix")))

    assert await storage.list_files("missing") == []


@pytest.mark.asyncio
async def test_backend_errors_propagate():
    storage = _storage(_FakeS3(glob_error=PermissionError("AccessDenied")))

    with pytest.raises(PermissionError):
        await storage.list_files("prefix")


@pytest.mark.asyncio
async def test_listing_returns_paths_relative_to_the_storage_root():
    storage = _storage(_FakeS3(paths=["bucket/root/prefix/a.txt", "bucket/root/prefix/b.txt"]))

    assert await storage.list_files("prefix") == ["prefix/a.txt", "prefix/b.txt"]
