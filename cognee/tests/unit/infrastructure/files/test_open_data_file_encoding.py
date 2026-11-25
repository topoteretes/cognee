"""``open_data_file`` forwards ``encoding`` to the local backend only.

``S3FileStorage.open(file_path, mode)`` takes no keyword arguments beyond ``mode``;
the S3 branch must therefore never receive ``encoding``, while the local backend
hands it to the builtin ``open()``. Regression for the S3 pipeline run that failed
with ``S3FileStorage.open() got an unexpected keyword argument 'encoding'``.
"""

import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest

import cognee.infrastructure.files.storage.S3FileStorage as s3_module
from cognee.infrastructure.files.utils.open_data_file import open_data_file


class _FakeS3FileStorage:
    """Same ``open`` signature as the real S3 backend: ``mode`` only, no kwargs."""

    calls: list[tuple[str, str]] = []

    def __init__(self, storage_path: str) -> None:
        self.storage_path = storage_path

    @asynccontextmanager
    async def open(self, file_path: str, mode: str = "r") -> AsyncGenerator:
        type(self).calls.append((file_path, mode))
        yield io.StringIO("from s3") if "b" not in mode else io.BytesIO(b"from s3")


@pytest.mark.asyncio
async def test_s3_branch_never_forwards_encoding(monkeypatch):
    monkeypatch.setattr(s3_module, "S3FileStorage", _FakeS3FileStorage)
    _FakeS3FileStorage.calls = []

    async with open_data_file("s3://bucket/dir/file.txt", mode="r", encoding="utf-8") as file:
        assert file.read() == "from s3"

    assert _FakeS3FileStorage.calls == [("file.txt", "r")]


@pytest.mark.asyncio
async def test_local_branch_forwards_encoding(tmp_path):
    target = tmp_path / "latin.txt"
    target.write_bytes("café".encode("latin-1"))

    async with open_data_file(str(target), mode="r", encoding="latin-1") as file:
        assert file.read() == "café"

    async with open_data_file(target.as_uri(), mode="r", encoding="latin-1") as file:
        assert file.read() == "café"
