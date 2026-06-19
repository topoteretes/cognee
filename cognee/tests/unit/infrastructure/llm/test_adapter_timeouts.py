import asyncio
import inspect
import threading
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral import (
    adapter as mistral_module,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral.adapter import (
    MistralAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.ollama.adapter import (
    OllamaAPIAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai import (
    adapter as openai_module,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
    OpenAIAdapter,
)


class ResponseModel(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_openai_transcription_uses_async_api_and_native_timeout(monkeypatch, tmp_path):
    captured = {}

    async def fake_transcription(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="transcript")

    monkeypatch.setattr(openai_module.litellm, "atranscription", fake_transcription)
    monkeypatch.setattr(
        openai_module,
        "get_llm_context_config",
        lambda: SimpleNamespace(llm_call_timeout_seconds=17.0),
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter.transcription_model = "whisper-1"
    adapter.api_key = "test-key"
    adapter.endpoint = "https://example.test"
    adapter.api_version = None

    result = await inspect.unwrap(OpenAIAdapter.create_transcript)(adapter, str(audio_path))

    assert result.text == "transcript"
    assert captured["timeout"] == 17.0


@pytest.mark.asyncio
async def test_mistral_transcription_uses_async_api_and_native_timeout(monkeypatch, tmp_path):
    captured = {}

    class Transcriptions:
        async def complete_async(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="transcript")

    monkeypatch.setattr(
        mistral_module,
        "get_llm_context_config",
        lambda: SimpleNamespace(llm_call_timeout_seconds=17.5),
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    adapter = MistralAdapter.__new__(MistralAdapter)
    adapter.transcription_model = "mistral/voxtral-mini"
    adapter.mistral_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=Transcriptions()))

    result = await inspect.unwrap(MistralAdapter.create_transcript)(adapter, str(audio_path))

    assert result.text == "transcript"
    assert captured["timeout_ms"] == 17500


@pytest.mark.asyncio
async def test_ollama_sync_completion_does_not_block_event_loop():
    call_thread = None

    def create(**kwargs):
        nonlocal call_thread
        call_thread = threading.get_ident()
        return ResponseModel(value="done")

    adapter = OllamaAPIAdapter.__new__(OllamaAPIAdapter)
    adapter.model = "ollama/test"
    adapter.llm_args = {}
    adapter.aclient = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    event_loop_thread = threading.get_ident()
    result = await inspect.unwrap(OllamaAPIAdapter.acreate_structured_output)(
        adapter, "input", "system", ResponseModel
    )

    assert result == ResponseModel(value="done")
    assert call_thread is not None
    assert call_thread != event_loop_thread
