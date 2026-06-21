import io

import pytest

from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage


class FakeS3File:
    def __init__(self, storage, path):
        self.storage = storage
        self.path = path
        self.data = b""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.storage.files[self.path] = self.data

    def write(self, data):
        self.data += data


class FakeS3:
    def __init__(self):
        self.files = {}

    def exists(self, path):
        return path in self.files

    def open(self, path, mode, **kwargs):
        return FakeS3File(self, path)


class NonSeekableBytesIO(io.BytesIO):
    def seek(self, *args, **kwargs):
        raise io.UnsupportedOperation("seek")


def make_storage(fake_s3):
    storage = object.__new__(S3FileStorage)
    storage.storage_path = "s3://bucket/root"
    storage.s3 = fake_s3
    return storage


@pytest.mark.asyncio
async def test_store_rewinds_seekable_stream():
    fake_s3 = FakeS3()
    storage = make_storage(fake_s3)
    stream = io.BytesIO(b"abcdef")
    stream.read(3)

    await storage.store("seekable.bin", stream, overwrite=True)

    assert fake_s3.files["bucket/root/seekable.bin"] == b"abcdef"


@pytest.mark.asyncio
async def test_store_writes_nonseekable_stream():
    fake_s3 = FakeS3()
    storage = make_storage(fake_s3)
    stream = NonSeekableBytesIO(b"abcdef")

    await storage.store("nonseekable.bin", stream, overwrite=True)

    assert fake_s3.files["bucket/root/nonseekable.bin"] == b"abcdef"
