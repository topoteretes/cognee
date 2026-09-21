"""Unit tests for cognee.modules.integrations.google_drive.client.

aiohttp is mocked at the session seam. What is under test is the part of this
module that is policy rather than plumbing: the byte ceiling on a file read,
which is the only thing standing between a multi-gigabyte Drive file and the
process memory.
"""

import logging
from unittest.mock import patch

import pytest

from cognee.modules.integrations.google_drive import client


class _FakeContent:
    def __init__(self, total_bytes: int, chunk_size: int):
        self._total = total_bytes
        self._chunk = chunk_size

    def iter_chunked(self, _requested_size):
        # Deliberately ignores the caller's size and yields our own, the way a
        # real stream is free to: the ceiling must hold for any chunking.
        async def _iter():
            sent = 0
            while sent < self._total:
                size = min(self._chunk, self._total - sent)
                sent += size
                yield b"a" * size

        return _iter()


class _FakeResponse:
    def __init__(self, total_bytes: int, chunk_size: int, status: int = 200):
        self.status = status
        self.content = _FakeContent(total_bytes, chunk_size)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeSession:
    def __init__(self, response):
        self._response = response

    def get(self, *_args, **_kwargs):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


def _patched(total_bytes: int, chunk_size: int, status: int = 200):
    response = _FakeResponse(total_bytes, chunk_size, status)
    return patch.object(client.aiohttp, "ClientSession", lambda **_kw: _FakeSession(response))


@pytest.mark.asyncio
async def test_a_huge_file_is_cut_at_exactly_the_ceiling():
    # Appending a whole chunk and only then noticing the ceiling lets the read
    # overshoot by almost a full chunk, which makes MAX_FILE_BYTES a claim the
    # code does not honour. With 64 KiB chunks a 1,000,000 byte ceiling used
    # to return 1,048,576.
    chunk = 64 * 1024
    with _patched(total_bytes=client.MAX_FILE_BYTES * 3, chunk_size=chunk):
        text = await client.download_file("ya29.token", "file-1")

    assert len(text.encode("utf-8")) == client.MAX_FILE_BYTES


@pytest.mark.asyncio
async def test_the_ceiling_holds_whatever_the_stream_chunks_at():
    # A chunk larger than the ceiling is the degenerate case: the very first
    # append would blow past it.
    with _patched(total_bytes=client.MAX_FILE_BYTES * 2, chunk_size=client.MAX_FILE_BYTES * 2):
        text = await client.download_file("ya29.token", "file-1")

    assert len(text.encode("utf-8")) == client.MAX_FILE_BYTES


@pytest.mark.asyncio
async def test_a_file_under_the_ceiling_is_returned_whole():
    with _patched(total_bytes=1234, chunk_size=64 * 1024):
        text = await client.download_file("ya29.token", "file-1")

    assert len(text.encode("utf-8")) == 1234


@pytest.mark.asyncio
async def test_a_non_200_names_the_operation_and_status_only():
    with (
        _patched(total_bytes=10, chunk_size=10, status=403),
        pytest.raises(RuntimeError) as error,
    ):
        await client.export_file("ya29.token", "file-1", "text/plain")

    message = str(error.value)
    assert "file export" in message
    assert "403" in message
    # The file id appears in shareable URLs and every failure here is logged.
    assert "file-1" not in message


@pytest.mark.asyncio
async def test_a_file_ending_exactly_at_the_ceiling_is_not_logged_as_truncated(caplog):
    # A chunk that lands exactly on the ceiling with nothing behind it is a
    # complete file, not a truncated one. A `len(chunk) >= remaining` check
    # cannot tell the two apart and used to log this case as truncated too.
    chunk = 100_000  # divides MAX_FILE_BYTES evenly, so a chunk can land
    # exactly on the boundary instead of only near it.
    with (
        _patched(total_bytes=client.MAX_FILE_BYTES, chunk_size=chunk),
        caplog.at_level(logging.INFO),
    ):
        text = await client.download_file("ya29.token", "file-1")

    assert len(text.encode("utf-8")) == client.MAX_FILE_BYTES
    assert "truncated" not in caplog.text


@pytest.mark.asyncio
async def test_a_file_landing_exactly_on_the_ceiling_with_more_behind_it_is_logged(caplog):
    # The genuinely ambiguous case: a chunk fills exactly to the ceiling, and
    # only the next pull from the stream proves there was more data all
    # along. This is what the extra loop iteration exists to resolve.
    chunk = 100_000
    with (
        _patched(total_bytes=client.MAX_FILE_BYTES + chunk, chunk_size=chunk),
        caplog.at_level(logging.INFO),
    ):
        text = await client.download_file("ya29.token", "file-1")

    assert len(text.encode("utf-8")) == client.MAX_FILE_BYTES
    assert "truncated" in caplog.text
