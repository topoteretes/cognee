"""Tests for the MCP-sampling LLM backend (issue #3644).

The adapter delegates completions to a host MCP session via
``sampling/createMessage``. Core behaviour is pinned with a mock session (no real
MCP host, no LLM_API_KEY): the mock is injected by patching
``get_sampling_session`` where the adapter calls it, so these tests run without
``mcp`` installed.

* structured output is produced from free-text replies by embedding the JSON
  Schema and validating the reply;
* a malformed reply triggers a repair retry rather than failing outright;
* ``str`` response models pass the text through untouched;
* the factory builds the adapter with no API key;
* when no host session is available, the adapter raises
  ``MCPSamplingUnavailableError`` instead of silently doing nothing.

The capability-gating tests additionally exercise the automatic per-request
resolution against the real MCP SDK, and skip when ``mcp`` is not installed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.exceptions import MCPSamplingUnavailableError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mcp_sampling import (
    adapter as adapter_module,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mcp_sampling.adapter import (
    MCPSamplingAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mcp_sampling.session_context import (
    get_sampling_session,
)


class _Person(BaseModel):
    name: str
    age: int


def _reply(text: str) -> SimpleNamespace:
    """Mimic an MCP ``CreateMessageResult`` (single ``content`` block)."""
    return SimpleNamespace(content=SimpleNamespace(type="text", text=text))


def _mock_session(*replies: str) -> AsyncMock:
    """A session whose ``create_message`` returns the given replies in order."""
    session = AsyncMock()
    session.create_message.side_effect = [_reply(r) for r in replies]
    return session


def _bind(monkeypatch, session) -> None:
    """Make the adapter see ``session`` (or ``None``) as the active host session."""
    monkeypatch.setattr(adapter_module, "get_sampling_session", lambda: session)


@pytest.mark.asyncio
async def test_structured_output_from_sampling_reply(monkeypatch):
    session = _mock_session('{"name": "Ada", "age": 36}')
    _bind(monkeypatch, session)

    adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=256)
    result = await adapter.acreate_structured_output(
        text_input="Ada Lovelace, 36",
        system_prompt="Extract the person.",
        response_model=_Person,
    )

    assert isinstance(result, _Person)
    assert result.name == "Ada" and result.age == 36
    # The JSON Schema must have been embedded in the system prompt sent to the host.
    kwargs = session.create_message.call_args.kwargs
    assert "JSON Schema" in kwargs["system_prompt"]
    assert kwargs["max_tokens"] == 256


@pytest.mark.asyncio
async def test_markdown_fenced_reply_is_parsed(monkeypatch):
    session = _mock_session('```json\n{"name": "Grace", "age": 45}\n```')
    _bind(monkeypatch, session)

    adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=128)
    result = await adapter.acreate_structured_output(
        text_input="Grace Hopper, 45", system_prompt="Extract.", response_model=_Person
    )

    assert result.name == "Grace" and result.age == 45


@pytest.mark.asyncio
async def test_invalid_reply_triggers_repair_retry(monkeypatch):
    session = _mock_session("not json at all", '{"name": "Alan", "age": 41}')
    _bind(monkeypatch, session)

    adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=128)
    result = await adapter.acreate_structured_output(
        text_input="Alan Turing, 41", system_prompt="Extract.", response_model=_Person
    )

    assert result.name == "Alan" and result.age == 41
    assert session.create_message.await_count == 2
    # The retry must carry the repair feedback, not just resend the original prompt.
    retry_kwargs = session.create_message.call_args_list[1].kwargs
    assert "did not validate" in retry_kwargs["system_prompt"]


@pytest.mark.asyncio
async def test_str_response_model_passes_text_through(monkeypatch):
    session = _mock_session("just some prose")
    _bind(monkeypatch, session)

    adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=64)
    result = await adapter.acreate_structured_output(
        text_input="say something", system_prompt="You are terse.", response_model=str
    )

    assert result == "just some prose"


@pytest.mark.asyncio
async def test_extra_kwargs_are_accepted(monkeypatch):
    # Interface parity: callers (e.g. the /v1/llm routes) forward extra params.
    session = _mock_session('{"name": "Edsger", "age": 72}')
    _bind(monkeypatch, session)

    adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=64)
    result = await adapter.acreate_structured_output(
        text_input="Edsger Dijkstra, 72",
        system_prompt="Extract.",
        response_model=_Person,
        temperature=0.2,
        top_p=0.9,
    )

    assert result.name == "Edsger" and result.age == 72


@pytest.mark.asyncio
async def test_gives_up_after_retries_exhausted(monkeypatch):
    session = _mock_session("nope", "still nope")
    _bind(monkeypatch, session)

    adapter = MCPSamplingAdapter(
        model="host-default", max_completion_tokens=64, structured_output_retries=2
    )
    with pytest.raises(ValueError, match="could not produce structured output"):
        await adapter.acreate_structured_output(
            text_input="x", system_prompt="Extract.", response_model=_Person
        )

    assert session.create_message.await_count == 2


@pytest.mark.asyncio
async def test_unsupported_response_model_raises_type_error(monkeypatch):
    session = _mock_session("{}")
    _bind(monkeypatch, session)

    adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=64)
    with pytest.raises(TypeError):
        await adapter.acreate_structured_output(
            text_input="x", system_prompt="Extract.", response_model=dict
        )
    session.create_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_raises_when_no_session_available(monkeypatch):
    _bind(monkeypatch, None)

    adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=64)
    with pytest.raises(MCPSamplingUnavailableError):
        await adapter.acreate_structured_output(
            text_input="x", system_prompt="Extract.", response_model=_Person
        )


@pytest.mark.asyncio
async def test_create_transcript_returns_none():
    adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=64)
    assert await adapter.create_transcript("audio.wav") is None


@pytest.mark.asyncio
async def test_transcribe_image_not_supported():
    adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=64)
    with pytest.raises(NotImplementedError):
        await adapter.transcribe_image("image.png")


def test_factory_builds_mcp_sampling_adapter_without_api_key():
    # The headline promise of #3644: this provider needs no LLM_API_KEY.
    from cognee.context_global_variables import llm_config as llm_config_ctx
    from cognee.infrastructure.llm.config import LLMConfig
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
        _API_KEY_REQUIRED_PROVIDERS,
        LLMProvider,
        _build_llm_client_cache_key,
        _get_llm_client_cached,
        _raise_for_missing_api_key,
    )

    assert LLMProvider("mcp-sampling") is LLMProvider.MCP_SAMPLING
    assert LLMProvider.MCP_SAMPLING not in _API_KEY_REQUIRED_PROVIDERS
    # Must not raise even with no key.
    _raise_for_missing_api_key(LLMProvider.MCP_SAMPLING, None, raise_api_key_error=True)

    config = LLMConfig(
        llm_provider="mcp-sampling", llm_model="host-default", llm_api_key=None, llm_endpoint=""
    )
    token = llm_config_ctx.set(config)
    # Keep the process-global adapter LRU cache hermetic, like the sibling
    # factory tests (test_llm_global_cache, test_stage_routing).
    _get_llm_client_cached.cache_clear()
    try:
        client = _get_llm_client_cached(_build_llm_client_cache_key(config, 256))
    finally:
        llm_config_ctx.reset(token)
        _get_llm_client_cached.cache_clear()

    assert isinstance(client, MCPSamplingAdapter)
    assert client.model == "host-default"
    assert client.max_completion_tokens == 256


def _request_ctx_token(session):
    """Bind ``session`` on the MCP SDK's per-request context; returns a reset token."""
    from mcp.server.lowlevel.server import request_ctx
    from mcp.shared.context import RequestContext

    return request_ctx.set(
        RequestContext(request_id="test", meta=None, session=session, lifespan_context=None)
    )


class _CapabilitySession:
    """Stand-in host session that grants or denies sampling like a real one."""

    def __init__(self, granted: bool, reply: str = '{"name": "Ada", "age": 36}') -> None:
        self._granted = granted
        self._reply = reply
        self.create_message_calls = 0
        self.last_create_message_kwargs: dict = {}

    def check_client_capability(self, capability) -> bool:
        return self._granted

    async def create_message(self, **kwargs):
        from mcp.types import CreateMessageResult, TextContent

        self.create_message_calls += 1
        self.last_create_message_kwargs = kwargs
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=self._reply),
            model="host",
        )


def test_auto_session_requires_granted_sampling_capability():
    # A per-request session whose client did NOT grant sampling is treated as
    # unavailable, so the adapter can raise a clear error rather than let the SDK
    # emit a raw protocol error.
    pytest.importorskip("mcp")
    from mcp.server.lowlevel.server import request_ctx

    token = _request_ctx_token(_CapabilitySession(granted=False))
    try:
        assert get_sampling_session() is None
    finally:
        request_ctx.reset(token)


def test_session_without_capability_method_is_used():
    # A stand-in session that lacks check_client_capability is assumed usable.
    pytest.importorskip("mcp")
    from mcp.server.lowlevel.server import request_ctx

    session = SimpleNamespace()  # no check_client_capability
    token = _request_ctx_token(session)
    try:
        assert get_sampling_session() is session
    finally:
        request_ctx.reset(token)


@pytest.mark.asyncio
async def test_ungranted_capability_raises_clear_error():
    pytest.importorskip("mcp")
    from mcp.server.lowlevel.server import request_ctx

    session = _CapabilitySession(granted=False)
    token = _request_ctx_token(session)
    try:
        adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=64)
        with pytest.raises(MCPSamplingUnavailableError):
            await adapter.acreate_structured_output(
                text_input="x", system_prompt="Extract.", response_model=_Person
            )
    finally:
        request_ctx.reset(token)

    # The clear error must fire before any sampling/createMessage is issued.
    assert session.create_message_calls == 0


@pytest.mark.asyncio
async def test_auto_session_used_when_sampling_granted():
    # Full automatic path against real SDK types: granted capability + a real
    # CreateMessageResult parsed into the response model.
    pytest.importorskip("mcp")
    from mcp.server.lowlevel.server import request_ctx
    from mcp.types import SamplingMessage

    session = _CapabilitySession(granted=True)
    token = _request_ctx_token(session)
    try:
        adapter = MCPSamplingAdapter(model="host-default", max_completion_tokens=128)
        result = await adapter.acreate_structured_output(
            text_input="Ada Lovelace, 36", system_prompt="Extract.", response_model=_Person
        )
    finally:
        request_ctx.reset(token)

    assert result.name == "Ada" and result.age == 36
    # Messages must be typed SamplingMessage objects: the real SDK validates them
    # before pydantic coercion (SEP-1577), so plain dicts crash create_message.
    messages = session.last_create_message_kwargs["messages"]
    assert isinstance(messages[0], SamplingMessage)
    assert messages[0].content.text == "Ada Lovelace, 36"
