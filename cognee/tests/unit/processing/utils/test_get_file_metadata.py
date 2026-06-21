import io

import pytest

from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata


class NonSeekableBytesIO(io.BytesIO):
    name = "stream.txt"

    def seekable(self):
        return False

    def seek(self, *args):
        raise io.UnsupportedOperation("seek")

    def tell(self):
        raise io.UnsupportedOperation("tell")


class OSErrorSeekBytesIO(io.BytesIO):
    def seek(self, *args):
        raise OSError("cannot seek")


@pytest.mark.asyncio
async def test_get_file_metadata_handles_non_seekable_stream():
    metadata = await get_file_metadata(NonSeekableBytesIO(b"hello"), name="stream.txt")

    assert metadata["name"] == "stream"
    assert metadata["mime_type"] == "text/plain"
    assert metadata["extension"] == "txt"
    assert metadata["content_hash"] == ""
    assert metadata["file_size"] == len(b"hello")


@pytest.mark.asyncio
async def test_get_file_metadata_handles_seek_oserror_without_stream_name():
    metadata = await get_file_metadata(OSErrorSeekBytesIO(b"hello"), name="stream.txt")

    assert metadata["name"] is None
    assert metadata["mime_type"] == "text/plain"
    assert metadata["extension"] == "txt"
    assert metadata["content_hash"] == ""
    assert metadata["file_size"] == len(b"hello")
