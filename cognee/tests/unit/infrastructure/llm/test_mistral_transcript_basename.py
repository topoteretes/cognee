"""Regression test for filename derivation in MistralAdapter.create_transcript.

`create_transcript` derived the API ``file_name`` with ``input.split("/")[-1]``.
On Windows, the audio path is a backslash path (e.g. ``C:\\audio\\clip.mp3``) with
no forward slashes, so the split sent the *entire path* as ``file_name`` to the
Mistral API instead of the basename.

The fix normalizes both ``\\`` and ``/`` before taking the basename. This test is
cross-platform strict: the Windows-style input contains backslashes regardless of
host, so a revert to splitting on ``/`` only fails on any OS.
"""

import io
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral import (
    adapter as mistral_adapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral.adapter import (
    MistralAdapter,
)


def _adapter_without_init(captured: dict) -> MistralAdapter:
    # Bypass __init__ (needs API keys / real clients); set only what
    # create_transcript touches.
    adapter = MistralAdapter.__new__(MistralAdapter)
    adapter.transcription_model = "voxtral-mini-latest"

    class _Transcriptions:
        def complete(self, model, file):
            captured["file_name"] = file["file_name"]
            return SimpleNamespace(text="transcribed")

    adapter.mistral_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=_Transcriptions())
    )
    return adapter


@asynccontextmanager
async def _fake_open_data_file(path, mode="rb"):
    yield io.BytesIO(b"audio-bytes")


@pytest.mark.asyncio
async def test_windows_audio_path_sends_basename_as_file_name():
    captured = {}
    adapter = _adapter_without_init(captured)

    with patch.object(mistral_adapter, "open_data_file", _fake_open_data_file):
        result = await adapter.create_transcript(r"C:\Users\me\audio\clip.mp3")

    assert captured["file_name"] == "clip.mp3"
    assert result.text == "transcribed"


@pytest.mark.asyncio
async def test_posix_audio_path_sends_basename_as_file_name():
    captured = {}
    adapter = _adapter_without_init(captured)

    with patch.object(mistral_adapter, "open_data_file", _fake_open_data_file):
        await adapter.create_transcript("/home/me/audio/clip.mp3")

    assert captured["file_name"] == "clip.mp3"
