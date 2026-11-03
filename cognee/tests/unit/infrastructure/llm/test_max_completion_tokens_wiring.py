"""The configured LLM_MAX_COMPLETION_TOKENS must reach the request.

BedrockAdapter, OllamaAPIAdapter and LlamaCppAPIAdapter stored ``max_completion_tokens``
in ``__init__`` but never put it on the outgoing request, so the provider default
applied instead of the configured cap.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.bedrock.adapter import (
    BedrockAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llama_cpp.adapter import (
    LlamaCppAPIAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.ollama.adapter import (
    OllamaAPIAdapter,
)

_LLM = "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm"


class _Resp(BaseModel):
    value: str


@asynccontextmanager
async def _null_rate_limiter():
    yield


def test_bedrock_request_includes_max_completion_tokens():
    adapter = BedrockAdapter(
        model="bedrock/anthropic.claude-3-sonnet",
        api_key="key",
        max_completion_tokens=777,
    )

    params = adapter._create_bedrock_request("hello", "system prompt", _Resp)

    assert params["max_completion_tokens"] == 777


def test_bedrock_llm_args_override_max_completion_tokens():
    adapter = BedrockAdapter(
        model="bedrock/anthropic.claude-3-sonnet",
        api_key="key",
        max_completion_tokens=777,
        llm_args={"max_completion_tokens": 55},
    )

    params = adapter._create_bedrock_request("hello", "system prompt", _Resp)

    assert params["max_completion_tokens"] == 55


@pytest.mark.asyncio
@pytest.mark.parametrize("response_model", [_Resp, str])
async def test_ollama_request_includes_max_tokens(response_model):
    adapter = OllamaAPIAdapter(
        endpoint="http://localhost:11434/v1",
        api_key="ollama",
        model="llama3",
        name="ollama",
        max_completion_tokens=321,
    )
    create = AsyncMock(return_value=Mock(choices=[Mock(message=Mock(content="ok"))]))
    adapter.aclient = Mock()
    adapter.aclient.chat.completions.create = create
    adapter.client = Mock()
    adapter.client.chat.completions.create = create

    with patch(f"{_LLM}.ollama.adapter.llm_rate_limiter_context_manager", _null_rate_limiter):
        await adapter.acreate_structured_output("hello", "system prompt", response_model)

    assert create.await_args.kwargs["max_tokens"] == 321


@pytest.mark.asyncio
async def test_llama_cpp_server_request_includes_max_tokens():
    adapter = LlamaCppAPIAdapter(
        name="llama_cpp",
        max_completion_tokens=654,
        endpoint="http://localhost:8080/v1",
        api_key="key",
        model="local",
    )
    create = AsyncMock(return_value=_Resp(value="ok"))
    adapter.aclient = Mock()
    adapter.aclient.chat.completions.create = create

    with patch(f"{_LLM}.llama_cpp.adapter.llm_rate_limiter_context_manager", _null_rate_limiter):
        await adapter.acreate_structured_output("hello", "system prompt", _Resp)

    assert create.await_args.kwargs["max_tokens"] == 654


@pytest.mark.asyncio
async def test_llama_cpp_llm_args_override_max_tokens():
    adapter = LlamaCppAPIAdapter(
        name="llama_cpp",
        max_completion_tokens=654,
        endpoint="http://localhost:8080/v1",
        api_key="key",
        model="local",
        llm_args={"max_tokens": 9},
    )
    create = AsyncMock(return_value=_Resp(value="ok"))
    adapter.aclient = Mock()
    adapter.aclient.chat.completions.create = create

    with patch(f"{_LLM}.llama_cpp.adapter.llm_rate_limiter_context_manager", _null_rate_limiter):
        await adapter.acreate_structured_output("hello", "system prompt", _Resp)

    assert create.await_args.kwargs["max_tokens"] == 9
