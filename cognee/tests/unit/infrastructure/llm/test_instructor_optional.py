"""instructor is an optional extra: only the paths that use it require it."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.llm.exceptions import InstructorNotInstalledError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm import (
    get_llm_client as legacy_factory,
)


def _legacy_config(provider: str) -> LLMConfig:
    return LLMConfig(llm_provider=provider, llm_model="model", llm_api_key="key", _env_file=None)


def test_building_the_legacy_client_without_the_package_raises():
    with (
        patch.object(legacy_factory, "instructor_installed", return_value=False),
        patch.object(
            legacy_factory, "get_llm_context_config", return_value=_legacy_config("openai")
        ),
        pytest.raises(InstructorNotInstalledError) as raised,
    ):
        legacy_factory.get_llm_client()

    assert 'pip install "cognee[instructor]"' in str(raised.value)


def test_mcp_sampling_does_not_need_the_package():
    # The MCP sampling adapter delegates completions to the host and never imports instructor.
    with (
        patch.object(legacy_factory, "instructor_installed", return_value=False),
        patch.object(
            legacy_factory, "get_llm_context_config", return_value=_legacy_config("mcp-sampling")
        ),
    ):
        client = legacy_factory.get_llm_client()

    assert type(client).__name__ == "MCPSamplingAdapter"


@pytest.mark.asyncio
@pytest.mark.parametrize("framework", ["baml", "litellm_native"])
@pytest.mark.parametrize("method", ["create_transcript", "transcribe_image"])
async def test_transcription_uses_the_native_client_without_the_package(framework, method):
    # BAML has no transcription of its own; it must not fall back to the legacy client.
    gateway = importlib.import_module("cognee.infrastructure.llm.LLMGateway")
    native_factory = importlib.import_module(
        "cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client"
    )
    native_client = MagicMock()
    setattr(native_client, method, AsyncMock(return_value="native"))
    legacy = MagicMock(name="legacy get_llm_client")

    with (
        patch.object(legacy_factory, "instructor_installed", return_value=False),
        patch.object(
            gateway,
            "get_llm_config",
            return_value=SimpleNamespace(structured_output_framework=framework),
        ),
        patch.object(native_factory, "get_native_client", return_value=native_client),
        patch.object(legacy_factory, "get_llm_client", legacy),
    ):
        result = await getattr(gateway.LLMGateway, method)("some/file.bin")

    assert result == "native"
    legacy.assert_not_called()
