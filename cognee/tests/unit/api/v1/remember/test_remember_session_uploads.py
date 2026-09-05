"""Session writes must store text that arrives as a multipart upload.

HTTP callers (CloudClient, the agent plugins, curl) can only send text to
``POST /api/v1/remember`` as a file part. Before this fix the session path
rendered that upload as the ``[UploadFile]`` placeholder and skipped it —
returning ``session_stored`` while storing nothing.
"""

import importlib
import io
from types import SimpleNamespace

import pytest
from starlette.datastructures import UploadFile

# The package re-exports remember() under the module's name, so import the
# module object explicitly to reach its private helpers.
remember_module = importlib.import_module("cognee.api.v1.remember.remember")


class _RecordingSessionManager:
    is_available = True

    def __init__(self):
        self.entries = []

    async def add_qa(self, **kwargs):
        self.entries.append(kwargs)


@pytest.fixture
def session_manager(monkeypatch):
    manager = _RecordingSessionManager()
    module = importlib.import_module("cognee.infrastructure.session.get_session_manager")
    monkeypatch.setattr(module, "get_session_manager", lambda: manager)
    return manager


_USER = SimpleNamespace(id="user-1")


@pytest.mark.asyncio
async def test_text_upload_is_stored_as_its_content(session_manager):
    upload = UploadFile(io.BytesIO(b"prefer tabs over spaces"), filename="text_abc.txt")

    await remember_module._add_to_session("s1", [upload], _USER)

    assert len(session_manager.entries) == 1
    assert session_manager.entries[0]["answer"] == "prefer tabs over spaces"
    assert session_manager.entries[0]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_plain_file_like_text_is_stored(session_manager):
    handle = io.BytesIO("über important".encode("utf-8"))

    await remember_module._add_to_session("s1", handle, _USER)

    assert [entry["answer"] for entry in session_manager.entries] == ["über important"]


@pytest.mark.asyncio
async def test_binary_upload_is_still_skipped(session_manager):
    upload = UploadFile(io.BytesIO(b"\x89PNG\r\n\x1a\n\xff\xfe"), filename="image.png")

    await remember_module._add_to_session("s1", [upload], _USER)

    assert session_manager.entries == []


@pytest.mark.asyncio
async def test_mixed_list_keeps_text_and_drops_placeholder_only_when_all_binary(session_manager):
    text_upload = UploadFile(io.BytesIO(b"first note"), filename="a.txt")

    await remember_module._add_to_session("s1", ["inline note", text_upload], _USER)

    assert session_manager.entries[0]["answer"] == "inline note\n\nfirst note"


@pytest.mark.asyncio
async def test_upload_is_rewound_after_read(session_manager):
    upload = UploadFile(io.BytesIO(b"rewind me"), filename="a.txt")

    await remember_module._add_to_session("s1", [upload], _USER)

    assert await upload.read() == b"rewind me"
