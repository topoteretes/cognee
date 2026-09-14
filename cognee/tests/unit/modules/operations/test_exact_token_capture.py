"""Exact (provider-billed) token capture for operation records.

The gateway reads ``result._raw_response.usage`` — attached by instructor,
and by the litellm_native adapter since SDK-399 — and passes the exact
counts to ``record_llm_call`` as overrides (``_exact_usage_from_result``,
from PR #4342 / CLO-434), so pipeline_runs records what the provider
billed (including hidden reasoning tokens) instead of a char/4 estimate.
No LLM, no network.
"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import (
    _exact_usage_from_result,
    _record_session_usage_after,
)
from cognee.modules.operations import operation_usage_scope


class _Answer(BaseModel):
    text: str


def _with_usage(result, prompt_tokens, completion_tokens):
    object.__setattr__(
        result,
        "_raw_response",
        SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        ),
    )
    return result


def test_extract_exact_usage_reads_attached_raw_response():
    result = _with_usage(_Answer(text="hi"), 789, 2961)
    assert _exact_usage_from_result(result) == (789, 2961)


def test_extract_exact_usage_absent_returns_none():
    assert _exact_usage_from_result(_Answer(text="hi")) == (None, None)
    assert _exact_usage_from_result("plain string output") == (None, None)


@pytest.mark.asyncio
async def test_gateway_feeds_exact_usage_to_operation_accumulator():
    """End-to-end at unit grain: billed counts land in the operation scope."""

    async def _fake_llm_call():
        return _with_usage(_Answer(text="x" * 400), 1121, 2140)

    with operation_usage_scope() as usage:
        result = await _record_session_usage_after(_fake_llm_call(), text_input="tiny")

    assert isinstance(result, _Answer)
    # Exact billed counts, not len("tiny")//4 or len(json)//4 estimates.
    assert usage.tokens_in == 1121
    assert usage.tokens_out == 2140


@pytest.mark.asyncio
async def test_gateway_falls_back_to_estimates_without_raw_response():
    """Plain-str outputs (connectivity test) and BAML keep the char/4 estimate."""

    async def _fake_llm_call():
        return "y" * 400  # 100 estimated output tokens

    with operation_usage_scope() as usage:
        await _record_session_usage_after(_fake_llm_call(), text_input="z" * 80)

    assert usage.tokens_in == 20  # 80 chars / 4
    assert usage.tokens_out == 100  # 400 chars / 4


def test_native_adapter_attaches_raw_response():
    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        _attach_raw_response,
    )

    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))
    instance = _attach_raw_response(_Answer(text="ok"), response)
    assert _exact_usage_from_result(instance) == (10, 20)
