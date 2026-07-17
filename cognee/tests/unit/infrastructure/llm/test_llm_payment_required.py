"""
Tests that LLMPaymentRequiredError is raised and not retried when an LLM provider
returns HTTP 402 Payment Required.
"""

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)


class _SimpleModel(BaseModel):
    value: str


def _make_payment_error(status_code: int = 402) -> Exception:
    """Build a generic exception carrying a status_code attribute, like most SDK errors."""
    exc = Exception("Payment required")
    exc.status_code = status_code
    return exc


# ---------------------------------------------------------------------------
# LLMPaymentRequiredError basics
# ---------------------------------------------------------------------------


def test_llm_payment_required_error_message():
    err = LLMPaymentRequiredError()
    assert "payment" in str(err).lower() or "budget" in str(err).lower()


def test_llm_payment_required_error_custom_message():
    err = LLMPaymentRequiredError("Custom message")
    assert "Custom message" in str(err)


# ---------------------------------------------------------------------------
# GenericAPIAdapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generic_adapter_raises_payment_required_on_402(monkeypatch):
    adapter = GenericAPIAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",
        max_completion_tokens=1024,
        name="test",
    )

    class FakeCompletions:
        async def create(self, **kwargs):
            raise _make_payment_error(402)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    adapter.aclient = FakeClient()

    with pytest.raises(LLMPaymentRequiredError):
        await adapter.acreate_structured_output("input", "system", _SimpleModel)


@pytest.mark.asyncio
async def test_generic_adapter_does_not_wrap_non_402_errors(monkeypatch):
    adapter = GenericAPIAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",
        max_completion_tokens=1024,
        name="test",
    )

    class FakeCompletions:
        async def create(self, **kwargs):
            raise _make_payment_error(500)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    adapter.aclient = FakeClient()

    with pytest.raises(Exception) as exc_info:
        await adapter.acreate_structured_output("input", "system", _SimpleModel)

    assert not isinstance(exc_info.value, LLMPaymentRequiredError)


# ---------------------------------------------------------------------------
# OpenAIAdapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_adapter_raises_payment_required_on_402(monkeypatch):
    import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter as generic_mod
    import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter as openai_mod
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
        OpenAIAdapter,
    )

    monkeypatch.setattr(generic_mod.instructor, "from_litellm", lambda *a, **kw: object())
    monkeypatch.setattr(openai_mod.instructor, "from_litellm", lambda *a, **kw: object())

    adapter = OpenAIAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",
        max_completion_tokens=1024,
    )

    class FakeCompletions:
        async def create(self, **kwargs):
            raise _make_payment_error(402)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    adapter.aclient = FakeClient()

    with pytest.raises(LLMPaymentRequiredError):
        await adapter.acreate_structured_output("input", "system", _SimpleModel)


# ---------------------------------------------------------------------------
# AnthropicAdapter — wraps the direct client call (requires anthropic package)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_adapter_raises_payment_required_on_402(monkeypatch):
    anthropic = pytest.importorskip("anthropic", reason="anthropic package not installed")
    import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter as generic_mod

    monkeypatch.setattr(generic_mod.instructor, "from_litellm", lambda *a, **kw: object())

    class FakeAsyncAnthropic:
        class messages:
            @staticmethod
            def create(*args, **kwargs):
                pass

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAsyncAnthropic)
    monkeypatch.setattr(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter.instructor.patch",
        lambda create, mode: object(),
    )

    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter import (
        AnthropicAdapter,
    )

    adapter = AnthropicAdapter(
        api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        max_completion_tokens=1024,
    )

    async def _raise_402(*args, **kwargs):
        raise _make_payment_error(402)

    adapter.aclient = _raise_402

    with pytest.raises(LLMPaymentRequiredError):
        await adapter.acreate_structured_output("input", "system", _SimpleModel)


# ---------------------------------------------------------------------------
# Retry exclusion: 402 should not be retried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generic_adapter_does_not_retry_402(monkeypatch):
    """Verify the adapter calls the LLM exactly once on a 402 (no retries)."""
    adapter = GenericAPIAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",
        max_completion_tokens=1024,
        name="test",
    )

    call_count = 0

    class FakeCompletions:
        async def create(self, **kwargs):
            nonlocal call_count
            call_count += 1
            raise _make_payment_error(402)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    adapter.aclient = FakeClient()

    with pytest.raises(LLMPaymentRequiredError):
        await adapter.acreate_structured_output("input", "system", _SimpleModel)

    assert call_count == 1, f"Expected exactly 1 call, got {call_count} (402 should not be retried)"
