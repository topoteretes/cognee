"""OllamaAPIAdapter uses a native async client — it must not block the event loop.

The adapter wraps an async ``AsyncOpenAI`` client (``instructor.from_openai``), so its
``async def`` methods ``await`` the client directly. (The previous implementation wrapped a
*synchronous* client and offloaded the blocking call via ``asyncio.to_thread``; with the
async client that offload is gone and the call must be awaited natively.)
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.ollama.adapter import (
    OllamaAPIAdapter,
)

_MODULE = (
    "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.ollama.adapter"
)


class _Resp(BaseModel):
    value: str


@asynccontextmanager
async def _null_rate_limiter():
    yield


@pytest.mark.asyncio
async def test_acreate_structured_output_awaits_async_client():
    adapter = OllamaAPIAdapter(
        endpoint="http://localhost:11434/v1",
        api_key="ollama",
        model="llama3",
        name="ollama",
        max_completion_tokens=128,
        num_ctx=4096,
    )

    sentinel = _Resp(value="ok")
    # Structured output goes through the instructor-wrapped async ``aclient`` and
    # is awaited natively, so the mock must be awaitable.
    adapter.aclient = Mock()
    adapter.aclient.chat.completions.create = AsyncMock(return_value=sentinel)

    with (
        patch(f"{_MODULE}.llm_rate_limiter_context_manager", _null_rate_limiter),
        patch(f"{_MODULE}.asyncio.to_thread", wraps=asyncio.to_thread) as to_thread_spy,
    ):
        result = await adapter.acreate_structured_output("hello", "system prompt", _Resp)

    # The async client is awaited directly...
    adapter.aclient.chat.completions.create.assert_awaited_once()
    call_kwargs = adapter.aclient.chat.completions.create.await_args.kwargs
    assert call_kwargs["extra_body"]["options"]["num_ctx"] == 4096
    # ...not offloaded to a worker thread, and its result is returned unchanged.
    assert not to_thread_spy.called, "structured call must be awaited natively, not offloaded"
    assert result is sentinel


@pytest.mark.asyncio
async def test_acreate_structured_output_does_not_override_user_num_ctx():
    adapter = OllamaAPIAdapter(
        endpoint="http://localhost:11434/v1",
        api_key="ollama",
        model="llama3",
        name="ollama",
        max_completion_tokens=128,
        num_ctx=4096,
        llm_args={"extra_body": {"options": {"num_ctx": 8192}}},
    )

    adapter.aclient = Mock()
    adapter.aclient.chat.completions.create = AsyncMock(return_value=_Resp(value="ok"))

    with patch(f"{_MODULE}.llm_rate_limiter_context_manager", _null_rate_limiter):
        await adapter.acreate_structured_output("hello", "system prompt", _Resp)

    call_kwargs = adapter.aclient.chat.completions.create.await_args.kwargs
    assert call_kwargs["extra_body"]["options"]["num_ctx"] == 8192
