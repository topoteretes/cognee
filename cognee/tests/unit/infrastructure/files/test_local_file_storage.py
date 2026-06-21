import io

import pytest

from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage


class NonSeekableBytesIO(io.BytesIO):
    def seek(self, *args, **kwargs):
        raise io.UnsupportedOperation("seek")


@pytest.mark.asyncio
async def test_store_rewinds_seekable_stream(tmp_path):
    storage = LocalFileStorage(str(tmp_path))
    stream = io.BytesIO(b"abcdef")
    stream.read(3)

    await storage.store("seekable.bin", stream, overwrite=True)

    assert (tmp_path / "seekable.bin").read_bytes() == b"abcdef"


@pytest.mark.asyncio
async def test_store_writes_nonseekable_stream(tmp_path):
    storage = LocalFileStorage(str(tmp_path))
    stream = NonSeekableBytesIO(b"abcdef")

    await storage.store("nonseekable.bin", stream, overwrite=True)

    assert (tmp_path / "nonseekable.bin").read_bytes() == b"abcdef"
