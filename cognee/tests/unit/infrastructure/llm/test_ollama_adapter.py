"""OllamaAPIAdapter must not block the event loop on its synchronous client.

The adapter wraps a synchronous ``OpenAI`` client (``instructor.from_openai``),
so calling ``.create()`` directly from an ``async def`` blocks the event loop for
the full LLM round-trip. The calls must be offloaded with ``asyncio.to_thread``.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import Mock, patch

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
async def test_acreate_structured_output_offloads_blocking_client_call():
    adapter = OllamaAPIAdapter(
        endpoint="http://localhost:11434/v1",
        api_key="ollama",
        model="llama3",
        name="ollama",
        max_completion_tokens=128,
    )

    sentinel = _Resp(value="ok")
    # The production client is synchronous; replace it with a sync mock so a
    # missing offload would run inline on the event loop.
    adapter.aclient = Mock()
    adapter.aclient.chat.completions.create = Mock(return_value=sentinel)

    with (
        patch(f"{_MODULE}.llm_rate_limiter_context_manager", _null_rate_limiter),
        patch(f"{_MODULE}.asyncio.to_thread", wraps=asyncio.to_thread) as to_thread_spy,
    ):
        result = await adapter.acreate_structured_output("hello", "system prompt", _Resp)

    # The blocking synchronous call is offloaded to a worker thread...
    assert to_thread_spy.called, "client call was not offloaded via asyncio.to_thread"
    assert to_thread_spy.call_args.args[0] is adapter.aclient.chat.completions.create
    # ...and its result is returned unchanged.
    assert result is sentinel
