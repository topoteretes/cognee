"""Tests for the MCP-sampling LLM backend (issue #3644).

The adapter delegates completions to a host MCP session via
``sampling/createMessage``. These tests use a mock session (no real MCP host and
no LLM_API_KEY) to pin the behaviour that matters:

* structured output is produced from free-text replies by embedding the JSON
  Schema and validating the reply;
* a malformed reply triggers a repair retry rather than failing outright;
* ``str`` response models pass the text through untouched;
* when no host session is bound, the adapter raises
  ``MCPSamplingUnavailableError`` instead of silently doing nothing.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.exceptions import MCPSamplingUnavailableError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mcp_sampling import (
    McpSamplingAdapter,
    reset_sampling_session,
    set_sampling_session,
)


class _Person(BaseModel):
    name: str
    age: int


def _reply(text: str) -> SimpleNamespace:
    """Mimic an MCP ``CreateMessageResult`` (``.content.text``)."""
    return SimpleNamespace(content=SimpleNamespace(type="text", text=text))


def _mock_session(*replies: str) -> AsyncMock:
    """A session whose ``create_message`` returns the given replies in order."""
    session = AsyncMock()
    session.create_message.side_effect = [_reply(r) for r in replies]
    return session


@pytest.mark.asyncio
async def test_structured_output_from_sampling_reply():
    session = _mock_session('{"name": "Ada", "age": 36}')
    token = set_sampling_session(session)
    try:
        adapter = McpSamplingAdapter(model="host-default", max_completion_tokens=256)
        result = await adapter.acreate_structured_output(
            text_input="Ada Lovelace, 36",
            system_prompt="Extract the person.",
            response_model=_Person,
        )
    finally:
        reset_sampling_session(token)

    assert isinstance(result, _Person)
    assert result.name == "Ada" and result.age == 36
    # The JSON Schema must have been embedded in the system prompt sent to the host.
    kwargs = session.create_message.call_args.kwargs
    assert "JSON Schema" in kwargs["system_prompt"]
    assert kwargs["max_tokens"] == 256


@pytest.mark.asyncio
async def test_markdown_fenced_reply_is_parsed():
    session = _mock_session('```json\n{"name": "Grace", "age": 45}\n```')
    token = set_sampling_session(session)
    try:
        adapter = McpSamplingAdapter(model="host-default", max_completion_tokens=128)
        result = await adapter.acreate_structured_output(
            text_input="Grace Hopper, 45", system_prompt="Extract.", response_model=_Person
        )
    finally:
        reset_sampling_session(token)

    assert result.name == "Grace" and result.age == 45


@pytest.mark.asyncio
async def test_invalid_reply_triggers_repair_retry():
    session = _mock_session("not json at all", '{"name": "Alan", "age": 41}')
    token = set_sampling_session(session)
    try:
        adapter = McpSamplingAdapter(model="host-default", max_completion_tokens=128)
        result = await adapter.acreate_structured_output(
            text_input="Alan Turing, 41", system_prompt="Extract.", response_model=_Person
        )
    finally:
        reset_sampling_session(token)

    assert result.name == "Alan" and result.age == 41
    assert session.create_message.await_count == 2


@pytest.mark.asyncio
async def test_str_response_model_passes_text_through():
    session = _mock_session("just some prose")
    token = set_sampling_session(session)
    try:
        adapter = McpSamplingAdapter(model="host-default", max_completion_tokens=64)
        result = await adapter.acreate_structured_output(
            text_input="say something", system_prompt="You are terse.", response_model=str
        )
    finally:
        reset_sampling_session(token)

    assert result == "just some prose"


@pytest.mark.asyncio
async def test_gives_up_after_retries_exhausted():
    session = _mock_session("nope", "still nope")
    token = set_sampling_session(session)
    try:
        adapter = McpSamplingAdapter(
            model="host-default", max_completion_tokens=64, structured_output_retries=2
        )
        with pytest.raises(ValueError, match="could not produce structured output"):
            await adapter.acreate_structured_output(
                text_input="x", system_prompt="Extract.", response_model=_Person
            )
    finally:
        reset_sampling_session(token)

    assert session.create_message.await_count == 2


@pytest.mark.asyncio
async def test_raises_when_no_session_available():
    # No session bound and not running inside an MCP server.
    adapter = McpSamplingAdapter(model="host-default", max_completion_tokens=64)
    with pytest.raises(MCPSamplingUnavailableError):
        await adapter.acreate_structured_output(
            text_input="x", system_prompt="Extract.", response_model=_Person
        )


@pytest.mark.asyncio
async def test_create_transcript_returns_none():
    adapter = McpSamplingAdapter(model="host-default", max_completion_tokens=64)
    assert await adapter.create_transcript("audio.wav") is None


@pytest.mark.asyncio
async def test_transcribe_image_not_supported():
    adapter = McpSamplingAdapter(model="host-default", max_completion_tokens=64)
    with pytest.raises(NotImplementedError):
        await adapter.transcribe_image("image.png")
