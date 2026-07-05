import asyncio

import httpx
import litellm
import pytest
from tenacity import retry, stop_after_attempt, wait_none

from cognee.infrastructure.llm.exceptions import LLMQuotaExceededError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.rate_limiter import (
    sleep_and_retry_async,
    sleep_and_retry_sync,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.retry_predicates import (
    is_quota_exceeded_error,
    retry_if_retryable_llm_error,
    should_retry_llm_error,
)


def _rate_limit_error(message: str, response: httpx.Response | None = None):
    return litellm.exceptions.RateLimitError(
        message,
        llm_provider="openai",
        model="openai/gpt-5-mini",
        response=response,
    )


def test_quota_exhaustion_is_detected_from_message():
    error = _rate_limit_error("429: You exceeded your current quota. Code: insufficient_quota.")

    assert is_quota_exceeded_error(error)


def test_quota_exhaustion_is_detected_from_response_body():
    class ProviderError(Exception):
        pass

    response = httpx.Response(
        status_code=429,
        json={
            "error": {
                "code": "insufficient_quota",
                "message": "Your billing quota has been exceeded.",
            }
        },
        request=httpx.Request("POST", "https://api.openai.test/v1/chat/completions"),
    )
    error = ProviderError("429 provider error")
    error.response = response

    assert is_quota_exceeded_error(error)


def test_transient_rate_limit_is_still_retryable():
    assert not is_quota_exceeded_error(_rate_limit_error("429: rate limit exceeded"))


def test_quota_exhaustion_is_not_retried():
    attempts = 0
    error = _rate_limit_error("429: quota_exceeded; billing limit reached")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_none(),
        retry=retry_if_retryable_llm_error,
        reraise=True,
    )
    def always_fails():
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(LLMQuotaExceededError) as error_info:
        always_fails()

    assert attempts == 1
    assert "quota or billing limit was exhausted" in str(error_info.value)


def test_project_quota_error_is_not_wrapped_again():
    assert not should_retry_llm_error(LLMQuotaExceededError())


def test_cancellation_is_not_retried():
    assert not should_retry_llm_error(asyncio.CancelledError())


def test_transient_rate_limit_still_retries_until_stop_condition():
    attempts = 0
    error = _rate_limit_error("429: rate limit exceeded")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_none(),
        retry=retry_if_retryable_llm_error,
        reraise=True,
    )
    def always_fails():
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(litellm.exceptions.RateLimitError):
        always_fails()

    assert attempts == 3


def test_sleep_and_retry_quota_exhaustion_is_not_retried():
    attempts = 0

    @sleep_and_retry_sync(max_retries=3, initial_backoff=0)
    def always_fails():
        nonlocal attempts
        attempts += 1
        raise Exception("429: insufficient_quota; billing limit reached")

    with pytest.raises(LLMQuotaExceededError):
        always_fails()

    assert attempts == 1


@pytest.mark.asyncio
async def test_async_sleep_and_retry_quota_exhaustion_is_not_retried():
    attempts = 0

    @sleep_and_retry_async(max_retries=3, initial_backoff=0)
    async def always_fails():
        nonlocal attempts
        attempts += 1
        raise Exception("429: insufficient_quota; billing limit reached")

    with pytest.raises(LLMQuotaExceededError):
        await always_fails()

    assert attempts == 1
