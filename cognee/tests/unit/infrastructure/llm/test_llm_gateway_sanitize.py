"""LLMGateway.acreate_structured_output must strip unpaired UTF-16 surrogates from both
text_input and system_prompt before delegating to ANY provider client (OpenAI, Anthropic,
Gemini, Ollama, ...) -- this is the single choke point every provider routes through, so
sanitizing here (rather than per-adapter) protects all of them, not just Ollama.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway, _strip_surrogates

_CONFIG_TARGET = "cognee.infrastructure.llm.LLMGateway.get_llm_config"
_CLIENT_TARGET = (
    "cognee.infrastructure.llm.structured_output_framework.litellm_instructor"
    ".llm.get_llm_client.get_llm_client"
)


class _Resp(BaseModel):
    value: str


def _mock_config():
    config = Mock()
    config.structured_output_framework = "instructor"
    return config


def test_strip_surrogates_decodes_valid_utf16_surrogate_pair():
    pair = chr(0xD83D) + chr(0xDE00)
    assert _strip_surrogates(pair) == "\U0001f600"
    mixed = f"café {pair} done"
    assert _strip_surrogates(mixed) == "café \U0001f600 done"


def test_strip_surrogates_replaces_lone_high_and_low_surrogates():
    lone_high = f"before {chr(0xD83D)} after"
    lone_low = f"before {chr(0xDE00)} after"
    cleaned_high = _strip_surrogates(lone_high)
    cleaned_low = _strip_surrogates(lone_low)
    assert "\ufffd" in cleaned_high
    assert "\ufffd" in cleaned_low
    assert "\ud83d" not in cleaned_high
    assert "\ude00" not in cleaned_low
    cleaned_high.encode("utf-8")
    cleaned_low.encode("utf-8")


@pytest.mark.asyncio
async def test_acreate_structured_output_strips_surrogates_for_every_provider():
    fake_client = Mock()
    fake_client.acreate_structured_output = AsyncMock(return_value=_Resp(value="ok"))

    poisoned_input = "some session text \udc8f more text"
    poisoned_system = "system prompt with \udc8f surrogate"

    with (
        patch(_CONFIG_TARGET, return_value=_mock_config()),
        patch(_CLIENT_TARGET, return_value=fake_client),
    ):
        await LLMGateway.acreate_structured_output(poisoned_input, poisoned_system, _Resp)

    call_kwargs = fake_client.acreate_structured_output.call_args.kwargs
    sent_input = call_kwargs["text_input"]
    sent_system = call_kwargs["system_prompt"]

    assert "\udc8f" not in sent_input
    assert "\udc8f" not in sent_system
    sent_input.encode("utf-8")
    sent_system.encode("utf-8")


@pytest.mark.asyncio
async def test_acreate_structured_output_leaves_normal_text_unchanged():
    fake_client = Mock()
    fake_client.acreate_structured_output = AsyncMock(return_value=_Resp(value="ok"))

    with (
        patch(_CONFIG_TARGET, return_value=_mock_config()),
        patch(_CLIENT_TARGET, return_value=fake_client),
    ):
        await LLMGateway.acreate_structured_output("hello world", "system prompt", _Resp)

    call_kwargs = fake_client.acreate_structured_output.call_args.kwargs
    assert call_kwargs["text_input"] == "hello world"
    assert call_kwargs["system_prompt"] == "system prompt"
