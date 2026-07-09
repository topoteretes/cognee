"""Shared tenacity retry policy for LLM structured-output calls.

Used by every structured-output framework — the litellm/instructor adapters and
the BAML integration alike. Each ``acreate_structured_output`` retries until BOTH
floors are satisfied: at least ``LLM_MIN_RETRY_ATTEMPTS`` attempts AND at least
``LLM_MIN_RETRY_SECONDS`` of elapsed wall-clock time.

``&`` builds tenacity's ``stop_all`` predicate, which stops only once *every*
sub-condition holds (``|`` / ``stop_any`` would stop at whichever floor is hit
first). The predicate is stateless — it reads everything off the per-call retry
state — so this single instance is safe to share across every retry decorator.

Which failures are retried at all is a separate concern (``llm_retry_condition``):
authentication, not-found, cancellation, payment/budget exhaustion, and
quota/billing exhaustion are terminal, while transient provider failures remain
retryable.
"""

import asyncio

import litellm
from tenacity import retry_if_exception, stop_after_attempt, stop_after_delay

from cognee.infrastructure.llm.exceptions import (
    LLMPaymentRequiredError,
    LLMQuotaExceededError,
)

# Minimum number of attempts before the call is allowed to give up.
LLM_MIN_RETRY_ATTEMPTS = 2
# Minimum elapsed seconds before the call is allowed to give up.
LLM_MIN_RETRY_SECONDS = 240

# Stop retrying only once BOTH the attempt floor AND the time floor are met.
llm_retry_stop_condition = stop_after_attempt(LLM_MIN_RETRY_ATTEMPTS) & stop_after_delay(
    LLM_MIN_RETRY_SECONDS
)

# Terminal quota/billing wordings — no retry can fix these. Kept narrow so
# transient per-minute rate limits stay retryable: the bare phrase "exceeded
# your current quota" is excluded because Gemini free tier uses it for
# recoverable limits (OpenAI's terminal case still matches "insufficient_quota").
_TERMINAL_QUOTA_PATTERNS = (
    "insufficient_quota",  # OpenAI / Azure OpenAI: billing quota exhausted
    "quota_exceeded",  # provider quota-exhaustion error code
    "billing hard limit",  # OpenAI: monthly hard limit reached
    "credit balance is too low",  # Anthropic: prepaid credits exhausted
    "out of credits",
)


def is_quota_or_billing_error(error: BaseException) -> bool:
    """True when the error, or its ``__cause__`` chain, reports quota/billing exhaustion.

    Walks ``__cause__`` (adapters/instructor wrap the provider error with
    ``raise ... from``) but not ``__context__``, so an unrelated error merely
    raised while handling a quota error is not misclassified.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if any(pattern in message for pattern in _TERMINAL_QUOTA_PATTERNS):
            return True
        current = current.__cause__
    return False


def should_retry_llm_exception(error: BaseException) -> bool:
    non_retryable: tuple[type[BaseException], ...] = (
        asyncio.CancelledError,
        LLMQuotaExceededError,
        LLMPaymentRequiredError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.AuthenticationError,
    )
    if isinstance(error, non_retryable):
        return False
    return not is_quota_or_billing_error(error)


def raise_if_quota_error(error: BaseException) -> None:
    """Re-raise quota/billing exhaustion as the actionable ``LLMQuotaExceededError``."""
    if isinstance(error, LLMQuotaExceededError):
        raise error
    if is_quota_or_billing_error(error):
        raise LLMQuotaExceededError(str(error)) from error


llm_retry_condition = retry_if_exception(should_retry_llm_exception)
