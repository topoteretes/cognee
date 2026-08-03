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
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, stop_after_delay

from cognee.infrastructure.llm.exceptions import (
    LLMPaymentRequiredError,
    LLMQuotaExceededError,
    is_budget_exhausted_error,
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
    "budget has been exceeded",  # LiteLLM proxy: virtual-key spend cap reached (CLO-409)
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


def _find_budget_error(error: BaseException) -> BaseException | None:
    """Return the innermost budget/payment-exhaustion exception wrapped by ``error``.

    Robust to ``instructor`` wrapping: after instructor exhausts its own retries it
    raises ``InstructorRetryException``, whose top-level ``status_code``/``response`` no
    longer describe the provider error. The raw provider 402/429 still survives on the
    tenacity ``RetryError``'s ``last_attempt`` and on each
    ``InstructorRetryException.failed_attempts[*].exception``, so the structured
    :func:`is_budget_exhausted_error` detector is run across all of them — not merely
    a ``str()`` match — so budget exhaustion is caught even if the wording changes.

    Returns the matched exception (so callers can preserve its original wording — e.g.
    LiteLLM's "Budget has been exceeded", which downstream status classification keys
    off), or ``None`` when no budget error is present.
    """
    seen: set[int] = set()

    def _walk(exc: BaseException | None) -> BaseException | None:
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            if is_budget_exhausted_error(exc):
                return exc
            # tenacity RetryError wraps the final failed attempt.
            last_attempt = getattr(exc, "last_attempt", None)
            if last_attempt is not None:
                found = _walk(getattr(last_attempt, "_exception", None))
                if found is not None:
                    return found
            # instructor records the raw provider error for every retry it made.
            for attempt in getattr(exc, "failed_attempts", None) or []:
                found = _walk(getattr(attempt, "exception", None))
                if found is not None:
                    return found
            exc = exc.__cause__
        return None

    return _walk(error)


def _is_budget_error(error: BaseException) -> bool:
    """True when ``error`` — or anything it wraps — signals budget/payment exhaustion."""
    return _find_budget_error(error) is not None


def _budget_retry_enabled() -> bool:
    """Whether budget/payment exhaustion may be retried (pay-as-you-go bounded retry).

    Off by default (``LLM_BUDGET_MAX_RETRY_ATTEMPTS == 0``): a spend-capped key fails
    fast so it cannot drive a retry storm (CLO-409). Pay-as-you-go tenants can opt in.
    """
    # Imported lazily: importing config at module top is circular.
    from cognee.infrastructure.llm.config import get_llm_config

    return get_llm_config().llm_budget_max_retry_attempts > 0


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
    # Budget/payment exhaustion is terminal: retrying a spend-capped key only produces
    # more proxy-rejected 429s (the CLO-409 retry storm). Opt-in bounded retry only.
    if _is_budget_error(error):
        return _budget_retry_enabled()
    return not is_quota_or_billing_error(error)


def raise_if_quota_error(error: BaseException) -> None:
    """Re-raise quota/billing/budget exhaustion as an actionable terminal error."""
    if isinstance(error, (LLMQuotaExceededError, LLMPaymentRequiredError)):
        raise error
    # Budget exhaustion becomes a 402-typed error, keeping the matched provider error's
    # original wording (e.g. LiteLLM's "Budget has been exceeded") — not the wrapper's —
    # so downstream status classification (pod insufficient-credits labelling) still matches.
    budget_error = _find_budget_error(error)
    if budget_error is not None:
        raise LLMPaymentRequiredError(str(budget_error)) from error
    if is_quota_or_billing_error(error):
        raise LLMQuotaExceededError(str(error)) from error


llm_retry_condition = retry_if_exception(should_retry_llm_exception)


def instructor_async_retrying(max_attempts: int) -> AsyncRetrying:
    """Inner-retry policy for instructor's ``max_retries`` (CLO-420).

    Instructor re-asks the model on validation errors; by default (an int
    ``max_retries``) it retries on *any* error, so a budget/quota rejection is
    hit ``max_attempts`` times per logical call. Passing this ``AsyncRetrying``
    instead re-asks on validation errors up to ``max_attempts`` but stops
    immediately on terminal budget/quota/auth errors — reusing the same
    ``llm_retry_condition`` as the outer adapter retry, so classification is
    consistent across both layers.

    Must NOT set ``reraise=True``: instructor catches tenacity's ``RetryError``
    and wraps it in ``InstructorRetryException`` (with ``failed_attempts``
    populated), which the adapters' fallback / content-policy handlers and the
    outer budget detection all depend on.
    """
    return AsyncRetrying(stop=stop_after_attempt(max_attempts), retry=llm_retry_condition)
