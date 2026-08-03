"""Structured-output retries must satisfy BOTH floors before giving up.

The ``acreate_structured_output`` adapters are decorated with the shared
``llm_retry_stop_condition`` (``stop_after_attempt(N) & stop_after_delay(S)``).
The ``&`` builds tenacity's ``stop_all`` predicate, which stops retrying only
once *every* sub-condition is satisfied — i.e. at least ``LLM_MIN_RETRY_ATTEMPTS``
attempts AND at least ``LLM_MIN_RETRY_SECONDS`` elapsed. (``|`` would be
``stop_any`` and would give up at whichever floor is hit first.)

These tests pin that AND behavior so a regression to ``|`` — or dropping either
floor — is caught.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError, LLMQuotaExceededError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
    OpenAIAdapter,
)
from cognee.infrastructure.llm.retry_config import (
    LLM_MIN_RETRY_ATTEMPTS,
    LLM_MIN_RETRY_SECONDS,
    is_quota_or_billing_error,
    llm_retry_stop_condition,
    raise_if_quota_error,
    should_retry_llm_exception,
)

_MODULE = (
    "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter"
)

MIN_ATTEMPTS = LLM_MIN_RETRY_ATTEMPTS
MIN_SECONDS = LLM_MIN_RETRY_SECONDS


class _Resp(BaseModel):
    value: str


@asynccontextmanager
async def _null_rate_limiter():
    """Stand in for ``llm_rate_limiter_context_manager`` so the test never throttles."""
    yield


def _build_adapter() -> OpenAIAdapter:
    # Constructs fully offline — ``instructor.from_litellm`` makes no network call.
    return OpenAIAdapter(api_key="test-key", model="gpt-4o-mini", max_completion_tokens=128)


# --------------------------------------------------------------------------- #
# 0. Quota/billing classification: exhausted quota is terminal, transient
#    provider failures (including per-minute rate limits) stay retryable.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("insufficient_quota: You exceeded your current quota."),
        RuntimeError({"error": {"code": "quota_exceeded", "type": "billing"}}),
        RuntimeError("Your credit balance is too low to access the API."),
    ],
)
def test_quota_and_billing_errors_are_terminal(error):
    assert is_quota_or_billing_error(error) is True
    assert should_retry_llm_exception(error) is False


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("temporary billing service unavailable"),
        RuntimeError("Rate limit exceeded, please try again in 20s."),
        # Gemini free tier returns this exact wording for recoverable per-minute
        # limits (status RESOURCE_EXHAUSTED); it must stay retryable.
        RuntimeError(
            "You exceeded your current quota, please check your plan and billing "
            "details. status: RESOURCE_EXHAUSTED"
        ),
        RuntimeError("boom"),
    ],
)
def test_transient_errors_stay_retryable(error):
    assert is_quota_or_billing_error(error) is False
    assert should_retry_llm_exception(error) is True


def test_quota_error_is_found_on_the_cause_chain():
    wrapper = RuntimeError("instructor retries exhausted")
    wrapper.__cause__ = RuntimeError("insufficient_quota")

    assert is_quota_or_billing_error(wrapper) is True
    assert should_retry_llm_exception(wrapper) is False


# --------------------------------------------------------------------------- #
# 1. The combined stop predicate in isolation: stop ONLY when both floors meet.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "attempts, elapsed, should_stop",
    [
        (MIN_ATTEMPTS - 1, 10, False),  # below both floors          -> keep retrying
        (MIN_ATTEMPTS - 1, MIN_SECONDS + 99, False),  # time met, attempts NOT met -> keep retrying
        (MIN_ATTEMPTS, MIN_SECONDS - 1, False),  # attempts met, time NOT met -> keep retrying
        (MIN_ATTEMPTS, MIN_SECONDS, True),  # both floors exactly met    -> stop
        (MIN_ATTEMPTS + 6, MIN_SECONDS + 99, True),  # both floors well past      -> stop
    ],
)
def test_combined_stop_requires_both_conditions(attempts, elapsed, should_stop):
    # Exercise the actual shared predicate the adapters are decorated with.
    state = SimpleNamespace(attempt_number=attempts, seconds_since_start=elapsed)
    assert llm_retry_stop_condition(state) is should_stop


# --------------------------------------------------------------------------- #
# 1b. A quota error through the real adapter is NOT retried: exactly one
#     attempt, then the provider error propagates (issue #3643).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_structured_output_does_not_retry_quota_errors():
    adapter = _build_adapter()
    adapter.aclient = Mock()
    adapter.aclient.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("insufficient_quota: exceeded your current quota")
    )

    with patch(f"{_MODULE}.llm_rate_limiter_context_manager", _null_rate_limiter):
        with pytest.raises(RuntimeError, match="insufficient_quota"):
            await adapter.acreate_structured_output("hi", "system", _Resp)

    assert adapter.aclient.chat.completions.create.await_count == 1


# --------------------------------------------------------------------------- #
# 1c. LLMGateway converts the raw provider error into the actionable
#     LLMQuotaExceededError — for every provider and framework, since all
#     structured-output calls flow through the gateway.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_llm_gateway_converts_quota_errors():
    # Resolve modules via sys.modules, not a dotted patch string: the llm package
    # re-exports the LLMGateway *class*, so "cognee.infrastructure.llm.LLMGateway"
    # can resolve to the class instead of the module (order-dependent across Python
    # versions). sys.modules is keyed by name and always gives the module.
    import sys
    import cognee.infrastructure.llm.LLMGateway  # noqa: F401 — ensure in sys.modules
    import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client  # noqa: E501,F401

    gateway_module = sys.modules["cognee.infrastructure.llm.LLMGateway"]
    get_llm_client_module = sys.modules[
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client"
    ]

    failing_client = Mock()
    failing_client.acreate_structured_output = AsyncMock(
        side_effect=RuntimeError("insufficient_quota: exceeded your current quota")
    )
    fake_config = SimpleNamespace(structured_output_framework="instructor", llm_model="gpt-4o-mini")

    with (
        patch.object(gateway_module, "get_llm_config", return_value=fake_config),
        patch.object(get_llm_client_module, "get_llm_client", return_value=failing_client),
    ):
        with pytest.raises(LLMQuotaExceededError, match="not retryable"):
            await gateway_module.LLMGateway.acreate_structured_output("hi", "system", _Resp)

    assert failing_client.acreate_structured_output.await_count == 1


# --------------------------------------------------------------------------- #
# 2. End-to-end through the real adapter with the LLM call mocked: a transient
#    failure that clears returns the mocked response. (tenacity consults `stop`
#    only after a *failure*, so a success short-circuits both floors.)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_structured_output_retries_then_returns_mocked_response():
    adapter = _build_adapter()
    sentinel = _Resp(value="ok")

    adapter.aclient = Mock()
    adapter.aclient.chat.completions.create = AsyncMock(
        side_effect=[RuntimeError("boom"), RuntimeError("boom"), sentinel]
    )

    async def _instant(_seconds):  # neutralize backoff sleeps
        return None

    with (
        patch(f"{_MODULE}.llm_rate_limiter_context_manager", _null_rate_limiter),
        patch("asyncio.sleep", _instant),
    ):
        result = await adapter.acreate_structured_output("hi", "system", _Resp)

    assert result is sentinel
    assert adapter.aclient.chat.completions.create.await_count == 3


# --------------------------------------------------------------------------- #
# 3. End-to-end with the LLM call always failing: reaching the attempt floor
#    is NOT enough to stop — the retry keeps going until the 240s floor is also
#    crossed. A fake clock advanced by each backoff makes this deterministic and
#    instant (no real waiting).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_structured_output_keeps_retrying_until_time_floor():
    adapter = _build_adapter()
    adapter.aclient = Mock()
    adapter.aclient.chat.completions.create = AsyncMock(side_effect=RuntimeError("always"))

    clock = [0.0]

    async def _advancing_sleep(seconds):
        # Tenacity's only sleeps are the inter-retry backoffs; advance a fake clock
        # by exactly the requested backoff so `seconds_since_start` grows without
        # any real waiting. The coroutine never truly suspends, so the event loop
        # never consults its own (patched) clock — only tenacity does.
        clock[0] += float(seconds)

    def _fake_monotonic():
        return clock[0]

    with (
        patch(f"{_MODULE}.llm_rate_limiter_context_manager", _null_rate_limiter),
        patch("asyncio.sleep", _advancing_sleep),
        patch("time.monotonic", _fake_monotonic),
    ):
        with pytest.raises(RuntimeError, match="always"):
            await adapter.acreate_structured_output("hi", "system", _Resp)

    attempts = adapter.aclient.chat.completions.create.await_count
    # With the (8, 128) backoff the attempt floor is reached at ~24s, yet the call
    # must keep retrying past it because the 240s floor is not yet met.
    assert attempts > MIN_ATTEMPTS, (
        f"gave up after {attempts} attempts; the {MIN_SECONDS}s floor must keep it "
        "retrying past the attempt floor — both conditions are required to stop"
    )
    # It only gave up once the elapsed-time floor was actually crossed.
    assert clock[0] >= MIN_SECONDS


# --------------------------------------------------------------------------- #
# 4. Budget/payment exhaustion is terminal by default (CLO-409): a spend-capped
#    LiteLLM key must fail fast, never feed the ~15x retry storm — even though
#    ``instructor`` wraps the raw 429 in ``InstructorRetryException`` before we
#    see it. Opt-in bounded retry (pay-as-you-go) is exercised separately.
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class _ProxyBudget429(Exception):
    """Mimics LiteLLM's proxy budget rejection: HTTP 429 + error.type budget_exceeded."""

    status_code = 429

    def __init__(self):
        super().__init__(
            "Litellm_proxyException - Budget has been exceeded! "
            "Current cost: 503.6, Max budget: 345.0"
        )
        self.response = _FakeResponse({"error": {"type": "budget_exceeded"}})


class _FailedAttempt:
    def __init__(self, exc):
        self.exception = exc


class _InstructorRetryLike(Exception):
    """Mimics instructor.InstructorRetryException: the raw error is on failed_attempts."""

    def __init__(self, exc):
        super().__init__("All retries exhausted")
        self.failed_attempts = [_FailedAttempt(exc)]


def _make_budget_errors():
    raw = _ProxyBudget429()
    return {
        "structured_429": raw,
        "instructor_wrapped": _InstructorRetryLike(raw),  # raw survives on failed_attempts
        "string_only": RuntimeError("Budget has been exceeded! Max budget: 345"),
        "http_402": type("PaymentRequired", (Exception,), {"status_code": 402})(),
    }


@pytest.mark.parametrize(
    "error", list(_make_budget_errors().values()), ids=list(_make_budget_errors())
)
def test_budget_errors_are_terminal_by_default(error):
    # Default config: llm_budget_max_retry_attempts == 0 -> never retry.
    assert should_retry_llm_exception(error) is False


def test_raise_if_quota_error_preserves_budget_marker_for_wrapped_error():
    # The pod classifies "insufficient_credits" by matching the budget marker in the
    # stored run error, so the INNER provider wording must survive the wrapper.
    wrapped = _InstructorRetryLike(_ProxyBudget429())
    with pytest.raises(LLMPaymentRequiredError) as excinfo:
        raise_if_quota_error(wrapped)
    assert "budget has been exceeded" in str(excinfo.value).lower()


def test_budget_error_is_retryable_when_payg_bounded_retry_enabled():
    # Pay-as-you-go tenants may opt in (auto-recharge can lift the cap mid-run).
    error = _InstructorRetryLike(_ProxyBudget429())
    fake_config = SimpleNamespace(llm_budget_max_retry_attempts=3)
    # _budget_retry_enabled imports get_llm_config lazily from the config module.
    with patch("cognee.infrastructure.llm.config.get_llm_config", return_value=fake_config):
        assert should_retry_llm_exception(error) is True


def test_transient_error_unaffected_by_budget_gate():
    assert should_retry_llm_exception(RuntimeError("connection reset by peer")) is True


# --------------------------------------------------------------------------- #
# 5. instructor_async_retrying (CLO-420): the AsyncRetrying handed to instructor's
#    `max_retries` re-asks transient/validation errors up to the cap, but stops
#    immediately on a terminal budget error (so a spend-capped key is hit once, not
#    ~MAX_RETRIES times, per logical call).
# --------------------------------------------------------------------------- #
def _attempts_until_stop(max_attempts, error) -> int:
    from cognee.infrastructure.llm.retry_config import instructor_async_retrying

    holder = {"n": 0}

    async def _run():
        async for attempt in instructor_async_retrying(max_attempts):
            with attempt:
                holder["n"] += 1
                raise error

    try:
        asyncio.run(_run())
    except Exception:
        pass  # tenacity raises RetryError once it stops; we only care about the count
    return holder["n"]


def test_instructor_retrying_stops_immediately_on_budget():
    assert _attempts_until_stop(3, _ProxyBudget429()) == 1


def test_instructor_retrying_reasks_transient_up_to_cap():
    # A non-terminal error (a validation re-ask looks the same to the predicate)
    # keeps retrying until the attempt cap.
    assert _attempts_until_stop(3, RuntimeError("boom")) == 3
