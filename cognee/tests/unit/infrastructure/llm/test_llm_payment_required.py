"""
Tests that LLMPaymentRequiredError is raised and not retried when an LLM provider
returns HTTP 402 Payment Required.
"""

from unittest.mock import patch

import pytest
from instructor.core import InstructorRetryException
from pydantic import BaseModel

from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)


def _wrapped_budget_error() -> InstructorRetryException:
    """The shape a LiteLLM-proxy spend cap actually arrives in: instructor wraps
    the provider rejection, so the budget wording only survives in str(error)."""
    return InstructorRetryException(
        "Budget has been exceeded! Current cost: 20.0, Max budget: 10.0",
        n_attempts=1,
        total_usage=0,
    )


def _policy_worded_error() -> InstructorRetryException:
    """A genuine content-policy rejection — must NOT be misclassified as budget."""
    return InstructorRetryException(
        "content management policy violation", n_attempts=1, total_usage=0
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

    # A non-402 error is retried, and the retry policy is
    # stop_after_attempt(2) & stop_after_delay(240): the AND means the 240s
    # floor is always paid. Unpatched, this one test ran 250s on every CI
    # copy of the unit suite. Same fake clock as test_structured_output_retry:
    # advance a counter by each backoff instead of sleeping.
    clock = [0.0]

    async def _advancing_sleep(seconds):
        clock[0] += float(seconds)

    def _fake_monotonic():
        return clock[0]

    with (
        patch("asyncio.sleep", _advancing_sleep),
        patch("time.monotonic", _fake_monotonic),
        pytest.raises(Exception) as exc_info,
    ):
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


# ---------------------------------------------------------------------------
# COG-6477 gap 1 — instructor wraps a budget rejection in InstructorRetryException,
# which several adapters caught in a clause that bare-`raise`s before the budget
# handler further down is ever reached. The rejection then escaped as the raw
# InstructorRetryException instead of the typed LLMPaymentRequiredError (402),
# and callers not routed through LLMGateway had to string-match the message
# themselves.
# ---------------------------------------------------------------------------


def _openai_adapter(monkeypatch, **kwargs):
    import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter as generic_mod
    import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter as openai_mod
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
        OpenAIAdapter,
    )

    monkeypatch.setattr(generic_mod.instructor, "from_litellm", lambda *a, **kw: object())
    monkeypatch.setattr(openai_mod.instructor, "from_litellm", lambda *a, **kw: object())
    return OpenAIAdapter(
        api_key="test-key", model="openai/gpt-5-mini", max_completion_tokens=1024, **kwargs
    )


def _fake_client(side_effect):
    calls = {"count": 0}

    class FakeCompletions:
        async def create(self, **kwargs):
            calls["count"] += 1
            raise side_effect(calls["count"])

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    return FakeClient(), calls


@pytest.mark.asyncio
async def test_openai_wrapped_budget_rejection_converts_without_fallback(monkeypatch):
    """No fallback configured: the bare `raise` clause is the only exit, so the
    budget check has to happen right there or the error escapes untyped."""
    adapter = _openai_adapter(monkeypatch)
    adapter.aclient, calls = _fake_client(lambda n: _wrapped_budget_error())

    with pytest.raises(LLMPaymentRequiredError) as exc_info:
        await adapter.acreate_structured_output("input", "system", _SimpleModel)

    assert calls["count"] == 1
    assert "Budget has been exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_plain_instructor_failure_without_fallback_is_not_misclassified(monkeypatch):
    """Sanity check the other direction: a non-budget failure must still escape
    as-is when there is nothing to classify. Proves raise_if_budget_exhausted is
    a true no-op here, not a change in behaviour for ordinary failures.

    A non-budget InstructorRetryException is legitimately retryable, so the
    outer @retry (stop_after_attempt(2) & stop_after_delay(240)) pays the 240s
    floor for real unless the clock is faked — same technique as
    test_generic_adapter_does_not_wrap_non_402_errors above.
    """
    adapter = _openai_adapter(monkeypatch)
    adapter.aclient, calls = _fake_client(lambda n: _policy_worded_error())

    clock = [0.0]

    async def _advancing_sleep(seconds):
        clock[0] += float(seconds)

    def _fake_monotonic():
        return clock[0]

    with (
        patch("asyncio.sleep", _advancing_sleep),
        patch("time.monotonic", _fake_monotonic),
        pytest.raises(InstructorRetryException),
    ):
        await adapter.acreate_structured_output("input", "system", _SimpleModel)

    # Retried (this is a transient-shaped failure, not a budget one), but never
    # converted to LLMPaymentRequiredError along the way.
    assert calls["count"] > 1


@pytest.mark.asyncio
async def test_openai_budget_rejection_still_tries_a_configured_fallback(monkeypatch):
    """The fallback carries a different key, so a budget cap on the primary key
    is exactly the case the fallback exists for. Classifying before the fallback
    decision would silently remove that failover — this proves it is not."""
    adapter = _openai_adapter(
        monkeypatch, fallback_model="openai/gpt-5-nano", fallback_api_key="fallback-key"
    )
    calls = {"count": 0}

    class FakeCompletionsWithFallback:
        async def create(self, **kwargs):
            calls["count"] += 1
            if kwargs.get("api_key") == "fallback-key":
                return "fallback answer"
            raise _wrapped_budget_error()

    class FakeChatWithFallback:
        completions = FakeCompletionsWithFallback()

    class FakeClientWithFallback:
        chat = FakeChatWithFallback()

    adapter.aclient = FakeClientWithFallback()

    result = await adapter.acreate_structured_output("input", "system", _SimpleModel)

    assert result == "fallback answer"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_openai_budget_rejection_converts_when_fallback_also_capped(monkeypatch):
    """Both keys are out of budget: the nested handler (after the fallback
    attempt) has to classify too, since it is the one that would otherwise
    misroute this to ContentPolicyFilterError."""
    adapter = _openai_adapter(
        monkeypatch, fallback_model="openai/gpt-5-nano", fallback_api_key="fallback-key"
    )
    adapter.aclient, calls = _fake_client(lambda n: _wrapped_budget_error())

    with pytest.raises(LLMPaymentRequiredError):
        await adapter.acreate_structured_output("input", "system", _SimpleModel)

    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_azure_managed_identity_wrapped_budget_rejection_converts(monkeypatch):
    """The managed-identity branch duplicates openai's exception handling shape
    rather than delegating to it, so it needs the same fix verified separately.
    Constructed via __new__ to avoid requiring the azure-identity package."""
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.azure_openai.adapter import (
        AzureOpenAIAdapter,
    )

    adapter = object.__new__(AzureOpenAIAdapter)
    adapter.use_managed_identity = True
    adapter.model = "azure/gpt-4o-mini"
    adapter.llm_args = {}
    adapter.fallback_model = None
    adapter.fallback_api_key = None
    adapter.aclient, calls = _fake_client(lambda n: _wrapped_budget_error())

    with pytest.raises(LLMPaymentRequiredError):
        await adapter.acreate_structured_output("input", "system", _SimpleModel)

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_bedrock_wrapped_budget_rejection_converts(monkeypatch):
    """Bedrock has no fallback path, so classification sits directly in the
    single except clause ahead of the content-policy branch."""
    import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.bedrock.adapter as bedrock_mod
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.bedrock.adapter import (
        BedrockAdapter,
    )

    monkeypatch.setattr(bedrock_mod.instructor, "from_litellm", lambda *a, **kw: object())

    adapter = BedrockAdapter(model="anthropic.claude-3-sonnet", api_key="test-key")
    adapter.aclient, calls = _fake_client(lambda n: _wrapped_budget_error())

    with pytest.raises(LLMPaymentRequiredError):
        await adapter.acreate_structured_output("input", "system", _SimpleModel)

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_gemini_wrapped_budget_rejection_converts_without_fallback(monkeypatch):
    """Gemini's structure differs from openai's: a non-policy-worded
    InstructorRetryException never reaches the fallback attempt regardless of
    budget, so classification can sit at the very top of the clause."""
    import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.gemini.adapter as gemini_mod
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.gemini.adapter import (
        GeminiAdapter,
    )

    monkeypatch.setattr(gemini_mod.instructor, "from_litellm", lambda *a, **kw: object())

    adapter = GeminiAdapter(
        api_key="test-key", model="gemini/gemini-2.0-flash-exp", max_completion_tokens=1024
    )
    adapter.aclient, calls = _fake_client(lambda n: _wrapped_budget_error())

    with pytest.raises(LLMPaymentRequiredError):
        await adapter.acreate_structured_output("input", "system", _SimpleModel)

    assert calls["count"] == 1


# ---------------------------------------------------------------------------
# COG-6477 gap 2 — the transcription and image decorators carried a private
# retry_if_not_exception_type tuple with no budget or quota classification, so
# a rejection that can never succeed still burned three attempts (~6s). They
# now share llm_retry_condition with every structured-output call in these
# adapters.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path, class_name, method_name",
    [
        ("openai.adapter", "OpenAIAdapter", "create_transcript"),
        ("generic_llm_api.adapter", "GenericAPIAdapter", "create_transcript"),
        ("generic_llm_api.adapter", "GenericAPIAdapter", "transcribe_image"),
        ("ollama.adapter", "OllamaAPIAdapter", "create_transcript"),
        ("ollama.adapter", "OllamaAPIAdapter", "transcribe_image"),
        ("mistral.adapter", "MistralAdapter", "create_transcript"),
    ],
)
def test_media_decorators_use_the_shared_retry_condition(module_path, class_name, method_name):
    """Structural check that catches drift immediately: resolved by name, so a
    rename or a reverted decorator swap fails here rather than hiding behind
    a skip for an adapter whose optional SDK is not installed."""
    import importlib

    from cognee.infrastructure.llm.retry_config import llm_retry_condition

    root = "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm"
    try:
        module = importlib.import_module(f"{root}.{module_path}")
    except ImportError as error:  # optional provider SDK not installed
        pytest.skip(f"{module_path} unavailable: {error}")

    assert hasattr(module, class_name), f"{module_path} has no {class_name}"
    method = getattr(module, class_name).__dict__[method_name]
    assert method.retry.retry is llm_retry_condition


@pytest.mark.asyncio
async def test_generic_adapter_create_transcript_does_not_retry_budget_rejection(
    monkeypatch, tmp_path
):
    """End-to-end confirmation for one representative site: a budget rejection
    on the transcription path costs one call, not the old three-attempt ladder.

    create_transcript has no try/except of its own — it calls litellm directly
    with no instructor wrapping and no type conversion — so the fix here is
    purely about the retry *count*, not about the exception type that escapes.
    That is Gap 2 in full: it stops the ladder, it does not add a 402 contract
    this call site never had.
    """
    adapter = GenericAPIAdapter(
        api_key="test-key", model="openai/gpt-5-mini", max_completion_tokens=1024, name="test"
    )

    audio_file = tmp_path / "input.mp3"
    audio_file.write_bytes(b"fake audio bytes")

    calls = {"count": 0}

    async def _raise_budget(**kwargs):
        calls["count"] += 1
        exc = Exception("Budget has been exceeded! Current cost: 20.0, Max budget: 10.0")
        exc.status_code = 402
        raise exc

    monkeypatch.setattr(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter.litellm.acompletion",
        _raise_budget,
    )

    with pytest.raises(Exception) as exc_info:
        await adapter.create_transcript(str(audio_file))

    assert not isinstance(exc_info.value, LLMPaymentRequiredError)
    assert calls["count"] == 1, f"Expected exactly 1 call, got {calls['count']}"
