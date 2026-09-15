"""LLMGateway must record REAL token counts when instructor makes them
available, instead of always falling back to the char-based estimate.

Instructor attaches the raw provider response as ``_raw_response`` on every
parsed model it returns (``instructor.processing.response.process_response``);
that raw response carries ``.usage`` with exact ``prompt_tokens`` /
``completion_tokens``. ``_exact_usage_from_result`` reads it, and
``_record_session_usage_after`` passes it through as
``tokens_in_override``/``tokens_out_override`` so ``record_llm_call`` never
has to guess for this path.
"""

import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import (
    _exact_usage_from_result,
    _record_session_usage_after,
)
from cognee.modules.session_lifecycle import usage_tracking
from cognee.modules.session_lifecycle.usage_tracking import track_session_usage

# `cognee.infrastructure.llm.__init__` does `from .LLMGateway import LLMGateway`,
# which overwrites the `LLMGateway` attribute on the package with the class —
# so `import cognee.infrastructure.llm.LLMGateway as x` binds `x` to the class,
# not the module. Go through sys.modules to get the actual module for patching.
llm_gateway_module = sys.modules["cognee.infrastructure.llm.LLMGateway"]


class _Answer(BaseModel):
    value: str


def _with_raw_response(model: BaseModel, prompt_tokens=None, completion_tokens=None) -> BaseModel:
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    model._raw_response = SimpleNamespace(usage=usage)
    return model


# --------------------------------------------------------------------------- #
# _exact_usage_from_result
# --------------------------------------------------------------------------- #


def test_exact_usage_extracted_from_raw_response():
    result = _with_raw_response(_Answer(value="hi"), prompt_tokens=123, completion_tokens=45)
    assert _exact_usage_from_result(result) == (123, 45)


def test_exact_usage_none_when_no_raw_response():
    # Plain string path (skips instructor) has no _raw_response attribute at all.
    assert _exact_usage_from_result("just a string") == (None, None)


def test_exact_usage_none_when_raw_response_has_no_usage():
    model = _Answer(value="hi")
    model._raw_response = SimpleNamespace()  # no .usage
    assert _exact_usage_from_result(model) == (None, None)


def test_exact_usage_none_when_usage_fields_missing():
    # e.g. a provider that returns a usage object without token fields.
    result = _with_raw_response(_Answer(value="hi"), prompt_tokens=None, completion_tokens=10)
    assert _exact_usage_from_result(result) == (None, None)


# --------------------------------------------------------------------------- #
# _record_session_usage_after end-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_record_session_usage_after_uses_exact_counts(monkeypatch):
    captured = {}

    async def fake_record_llm_call(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(usage_tracking, "record_llm_call", fake_record_llm_call)
    monkeypatch.setattr(
        llm_gateway_module,
        "get_llm_context_config",
        lambda: SimpleNamespace(llm_model="openai/gpt-4o-mini"),
    )

    async def coro():
        return _with_raw_response(_Answer(value="hi"), prompt_tokens=200, completion_tokens=50)

    user_id = uuid4()
    async with track_session_usage("sess-1", user_id):
        await _record_session_usage_after(coro(), text_input="some long prompt text")

    assert captured["tokens_in_override"] == 200
    assert captured["tokens_out_override"] == 50


@pytest.mark.asyncio
async def test_record_session_usage_after_falls_back_to_estimate_without_raw_response(
    monkeypatch,
):
    captured = {}

    async def fake_record_llm_call(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(usage_tracking, "record_llm_call", fake_record_llm_call)
    monkeypatch.setattr(
        llm_gateway_module,
        "get_llm_context_config",
        lambda: SimpleNamespace(llm_model="openai/gpt-4o-mini"),
    )

    async def coro():
        return "a plain string response, no instructor involved"

    user_id = uuid4()
    async with track_session_usage("sess-2", user_id):
        await _record_session_usage_after(coro(), text_input="prompt")

    # No exact usage available -> overrides are None, letting record_llm_call
    # fall back to its own char-based estimate.
    assert captured["tokens_in_override"] is None
    assert captured["tokens_out_override"] is None
