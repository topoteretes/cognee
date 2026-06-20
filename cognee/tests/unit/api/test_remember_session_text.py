"""Regression tests for #2888.

`remember(session_id=...)` over HTTP/MCP silently no-op'd: the client wraps the
payload as a multipart file upload, and the session-storage path used to coerce
file uploads to a placeholder string and skip the write. These tests pin that a
file-upload payload is now read as text and stored in the session cache.
"""

import io

import pytest

from cognee.api.v1.remember.remember import _add_to_session, _coerce_session_text


class _AsyncUpload:
    """Minimal stand-in for starlette ``UploadFile`` (async read/seek)."""

    def __init__(self, content: bytes, name: str = "upload.txt"):
        self._buffer = io.BytesIO(content)
        self.name = name

    async def read(self, *args):
        return self._buffer.read()

    async def seek(self, position):
        return self._buffer.seek(position)


class _FakeSessionManager:
    def __init__(self):
        self.is_available = True
        self.calls = []

    async def add_qa(self, **kwargs):
        self.calls.append(kwargs)
        return "qa-id"


class _User:
    id = "11111111-1111-1111-1111-111111111111"


@pytest.mark.asyncio
async def test_coerce_plain_string():
    assert await _coerce_session_text("hello world") == "hello world"


@pytest.mark.asyncio
async def test_coerce_bytes():
    assert await _coerce_session_text(b"hi there") == "hi there"


@pytest.mark.asyncio
async def test_coerce_sync_file_like_restores_position():
    file_like = io.BytesIO(b"session text content")
    assert await _coerce_session_text(file_like) == "session text content"
    # The stream position is restored so downstream readers see the full content.
    assert file_like.tell() == 0


@pytest.mark.asyncio
async def test_coerce_async_upload_file():
    upload = _AsyncUpload(b"uploaded session text")
    assert await _coerce_session_text(upload) == "uploaded session text"


@pytest.mark.asyncio
async def test_coerce_list_of_uploads():
    uploads = [_AsyncUpload(b"alpha"), _AsyncUpload(b"beta")]
    assert await _coerce_session_text(uploads) == "alpha\n\nbeta"


@pytest.mark.asyncio
async def test_coerce_binary_payload_yields_empty():
    # Non-UTF-8 bytes are treated as non-text; caller skips the write.
    assert await _coerce_session_text(io.BytesIO(b"\xff\xfe\x00\x01")) == ""


@pytest.mark.asyncio
async def test_add_to_session_stores_uploaded_file_text(monkeypatch):
    """The core #2888 regression: a file-upload payload must be stored."""
    import sys
    import cognee.infrastructure.session.get_session_manager  # noqa: F401  (ensure loaded)

    module = sys.modules["cognee.infrastructure.session.get_session_manager"]
    fake_sm = _FakeSessionManager()
    monkeypatch.setattr(module, "get_session_manager", lambda: fake_sm)

    upload = _AsyncUpload(b"remember this for the session")
    await _add_to_session("sess-123", upload, _User())

    assert len(fake_sm.calls) == 1
    stored = fake_sm.calls[0]
    assert stored["session_id"] == "sess-123"
    assert stored["answer"] == "remember this for the session"
    assert stored["user_id"] == _User.id


@pytest.mark.asyncio
async def test_add_to_session_skips_empty_payload(monkeypatch):
    import sys
    import cognee.infrastructure.session.get_session_manager  # noqa: F401  (ensure loaded)

    module = sys.modules["cognee.infrastructure.session.get_session_manager"]
    fake_sm = _FakeSessionManager()
    monkeypatch.setattr(module, "get_session_manager", lambda: fake_sm)

    await _add_to_session("sess-123", io.BytesIO(b""), _User())

    assert fake_sm.calls == []
