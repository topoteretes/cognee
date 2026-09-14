"""Audio and image transcription on the litellm_native path.

Only the litellm network boundary (``atranscription`` / ``acompletion``) is mocked, so the
adapter's real request construction — file handling, MIME detection, message assembly,
kwarg passthrough — is exercised. Also pins that ``LLMGateway`` routes both entry points
to the native client when ``litellm_native`` is configured and to the legacy client
otherwise.
"""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
    NativeLiteLLMAdapter,
)
from cognee.infrastructure.llm.types import TranscriptionReturnType

TEST_DATA = Path(__file__).parents[3] / "test_data"
AUDIO = TEST_DATA / "text_to_speech.mp3"
IMAGE = TEST_DATA / "revenue_table.png"


def _adapter(**overrides) -> NativeLiteLLMAdapter:
    params = {
        "api_key": "test-key",
        "model": "openai/gpt-5-mini",
        "max_completion_tokens": 1024,
        "endpoint": "https://example.test/v1",
        "api_version": "2024-01-01",
        "transcription_model": "whisper-1",
    }
    params.update(overrides)
    return NativeLiteLLMAdapter(**params)


def _content_part(kwargs, part_type):
    parts = kwargs["messages"][0]["content"]
    return next(part for part in parts if part["type"] == part_type)


# ---- create_transcript ----


@pytest.mark.asyncio
async def test_create_transcript_uses_litellm_atranscription_and_passes_kwargs():
    response = SimpleNamespace(text="hello world", segments=[{"start": 0.0, "text": "hello"}])
    fake = AsyncMock(return_value=response)

    with patch("litellm.atranscription", fake):
        result = await _adapter().create_transcript(
            str(AUDIO), response_format="verbose_json", timestamp_granularities=["segment"]
        )

    kwargs = fake.call_args.kwargs
    assert kwargs["model"] == "whisper-1"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["api_base"] == "https://example.test/v1"
    assert kwargs["api_version"] == "2024-01-01"
    assert kwargs["response_format"] == "verbose_json"
    assert kwargs["timestamp_granularities"] == ["segment"]
    assert hasattr(kwargs["file"], "read")

    assert isinstance(result, TranscriptionReturnType)
    assert result.text == "hello world"
    assert result.payload is response  # the video loader reads segments off the payload


@pytest.mark.asyncio
async def test_create_transcript_defaults_transcription_model_to_chat_model():
    fake = AsyncMock(return_value=SimpleNamespace(text="x"))

    with patch("litellm.atranscription", fake):
        await _adapter(transcription_model=None).create_transcript(str(AUDIO))

    assert fake.call_args.kwargs["model"] == "openai/gpt-5-mini"


@pytest.mark.asyncio
async def test_create_transcript_rejects_response_without_text():
    fake = AsyncMock(return_value=SimpleNamespace())

    with patch("litellm.atranscription", fake), pytest.raises(ValueError, match="No text"):
        await _adapter().create_transcript.retry_with(stop=lambda _: True)(_adapter(), str(AUDIO))


# ---- transcribe_image ----


@pytest.mark.asyncio
async def test_transcribe_image_builds_multimodal_request():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    fake = AsyncMock(return_value=response)

    with patch("litellm.acompletion", fake):
        result = await _adapter().transcribe_image(
            str(IMAGE), prompt="EXTRACT ENTITIES", max_completion_tokens=777, reasoning_effort="low"
        )

    kwargs = fake.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-5-mini"
    assert _content_part(kwargs, "text")["text"] == "EXTRACT ENTITIES"
    assert _content_part(kwargs, "image_url")["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert kwargs["max_completion_tokens"] == 777
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["drop_params"] is True
    # Loaders read the caption off the raw ModelResponse.
    assert result.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_transcribe_image_defaults_are_backwards_compatible():
    fake = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
    )

    with patch("litellm.acompletion", fake):
        await _adapter().transcribe_image(str(IMAGE))

    kwargs = fake.call_args.kwargs
    assert _content_part(kwargs, "text")["text"] == "What's in this image?"
    assert kwargs["max_completion_tokens"] == 300


@pytest.mark.asyncio
async def test_transcribe_image_rejects_non_image_mime(tmp_path):
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_bytes(b"plain text")
    fake = AsyncMock()

    with patch("litellm.acompletion", fake), pytest.raises(ValueError, match="MIME type"):
        await _adapter().transcribe_image.retry_with(stop=lambda _: True)(
            _adapter(), str(not_an_image)
        )

    fake.assert_not_called()


# ---- factory wiring ----


def test_get_native_client_passes_transcription_model():
    native_factory = importlib.import_module(
        "cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client"
    )
    config = SimpleNamespace(
        llm_api_key="test-key",
        llm_provider="openai",
        llm_azure_use_managed_identity=False,
        llm_model="gpt-5-mini",
        llm_max_completion_tokens=4096,
        llm_endpoint="",
        llm_api_version=None,
        fallback_model="",
        fallback_api_key="",
        fallback_endpoint="",
        llm_args={},
        transcription_model="whisper-1",
    )

    with patch.object(native_factory, "get_llm_context_config", return_value=config):
        client = native_factory.get_native_client()

    assert client.transcription_model == "whisper-1"


# ---- LLMGateway routing ----


def _gateway_and_factories():
    gateway = importlib.import_module("cognee.infrastructure.llm.LLMGateway")
    native_factory = importlib.import_module(
        "cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client"
    )
    legacy_factory = importlib.import_module(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client"
    )
    return gateway, native_factory, legacy_factory


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["create_transcript", "transcribe_image"])
async def test_gateway_routes_transcription_to_native_client(method):
    gateway, native_factory, legacy_factory = _gateway_and_factories()
    native_client = MagicMock()
    setattr(native_client, method, AsyncMock(return_value="native"))
    legacy = MagicMock(name="legacy get_llm_client")
    config = SimpleNamespace(structured_output_framework="litellm_native")

    with (
        patch.object(gateway, "get_llm_config", return_value=config),
        patch.object(native_factory, "get_native_client", return_value=native_client),
        patch.object(legacy_factory, "get_llm_client", legacy),
    ):
        result = await getattr(gateway.LLMGateway, method)("some/file.bin")

    assert result == "native"
    legacy.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("framework", ["instructor", "baml"])
async def test_gateway_keeps_legacy_client_for_opt_in_frameworks(framework):
    gateway, native_factory, legacy_factory = _gateway_and_factories()
    legacy_client = MagicMock()
    legacy_client.create_transcript = AsyncMock(return_value="legacy")
    native = MagicMock(name="get_native_client")
    config = SimpleNamespace(structured_output_framework=framework)

    with (
        patch.object(gateway, "get_llm_config", return_value=config),
        patch.object(native_factory, "get_native_client", native),
        patch.object(legacy_factory, "get_llm_client", return_value=legacy_client),
    ):
        result = await gateway.LLMGateway.create_transcript("some/file.mp3")

    assert result == "legacy"
    native.assert_not_called()
