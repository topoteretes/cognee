"""Shared tenacity retry policy for LLM structured-output calls.

Structured-output calls keep the existing stop policy: retry until both the
minimum attempt count and minimum elapsed time are reached. Retry selection is
separate: authentication, not-found, cancellation, and quota/billing exhaustion
are terminal; transient provider failures remain retryable.
"""

import asyncio
from collections.abc import Iterable
from typing import Any

from tenacity import retry_if_exception, stop_after_attempt, stop_after_delay

from cognee.infrastructure.llm.exceptions import LLMQuotaExceededError

try:
    import litellm
except Exception:  # pragma: no cover - litellm is an optional import in some test slices
    litellm = None


# Minimum number of attempts before the call is allowed to give up.
LLM_MIN_RETRY_ATTEMPTS = 2
# Minimum elapsed seconds before the call is allowed to give up.
LLM_MIN_RETRY_SECONDS = 240

_TERMINAL_QUOTA_PATTERNS = (
    "insufficient_quota",
    "quota_exceeded",
    "quota exceeded",
    "quota has been exceeded",
    "exceeded your current quota",
    "exceeded current quota",
    "429-billing",
    "billing limit",
    "billing hard limit",
    "billing quota",
    "payment required",
    "out of credits",
    "credit balance",
    "hard limit",
)


# Stop retrying only once BOTH the attempt floor AND the time floor are met.
llm_retry_stop_condition = stop_after_attempt(LLM_MIN_RETRY_ATTEMPTS) & stop_after_delay(
    LLM_MIN_RETRY_SECONDS
)


def _non_retryable_exception_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [asyncio.CancelledError, LLMQuotaExceededError]
    if litellm is not None:
        types.extend(
            [
                litellm.exceptions.NotFoundError,
                litellm.exceptions.AuthenticationError,
            ]
        )
    return tuple(types)


def _iter_error_values(value: Any, seen: set[int] | None = None) -> Iterable[str]:
    if value is None:
        return
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, (str, int, float, bool)):
        yield str(value)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_error_values(key, seen)
            yield from _iter_error_values(item, seen)
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_error_values(item, seen)
        return

    yield str(value)

    for attr in (
        "code",
        "type",
        "message",
        "body",
        "response",
        "json_body",
        "status_code",
        "llm_provider",
    ):
        try:
            yield from _iter_error_values(getattr(value, attr), seen)
        except Exception:
            continue

    if isinstance(value, BaseException):
        for arg in value.args:
            yield from _iter_error_values(arg, seen)
        yield from _iter_error_values(value.__cause__, seen)
        yield from _iter_error_values(value.__context__, seen)


def is_quota_or_billing_error(error: BaseException) -> bool:
    for value in _iter_error_values(error):
        lowered = value.lower()
        if any(pattern in lowered for pattern in _TERMINAL_QUOTA_PATTERNS):
            return True
    return False


def should_retry_llm_exception(error: BaseException) -> bool:
    if isinstance(error, _non_retryable_exception_types()):
        return False
    if is_quota_or_billing_error(error):
        return False
    return True


def raise_if_non_retryable_llm_error(error: BaseException) -> None:
    if isinstance(error, LLMQuotaExceededError):
        raise error
    if is_quota_or_billing_error(error):
        raise LLMQuotaExceededError(str(error)) from error


llm_retry_condition = retry_if_exception(should_retry_llm_exception)
