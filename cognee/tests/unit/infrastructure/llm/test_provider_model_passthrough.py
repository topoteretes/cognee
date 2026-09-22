from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from cognee.context_global_variables import llm_config as llm_config_context
from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.llm.structured_output_framework.litellm_native import native_adapter
from cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client import (
    get_native_client,
)


class ExtractedEntity(BaseModel):
    name: str


@pytest.mark.asyncio
async def test_anthropic_environment_config_reaches_provider_without_model_allowlist(monkeypatch):
    model_id = "anthropic/claude-sonnet-5"
    api_key = "provider-api-key"
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", model_id)
    monkeypatch.setenv("LLM_API_KEY", api_key)

    config = LLMConfig(_env_file=None)
    config_token = llm_config_context.set(config)
    mock_completion = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"name":"Cognee"}'))]
        )
    )
    monkeypatch.setattr(native_adapter.litellm, "acompletion", mock_completion)
    monkeypatch.setattr(native_adapter.litellm, "supports_response_schema", lambda **_: False)

    try:
        client = get_native_client()
        result = await client.acreate_structured_output(
            text_input="Cognee is a memory platform.",
            system_prompt="Extract the entity.",
            response_model=ExtractedEntity,
        )
    finally:
        llm_config_context.reset(config_token)

    assert result == ExtractedEntity(name="Cognee")
    assert mock_completion.await_args.kwargs["model"] == model_id
    assert mock_completion.await_args.kwargs["api_key"] == api_key
