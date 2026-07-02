import asyncio
from collections.abc import Iterator
from typing import Any

import litellm
from tenacity import retry_if_exception

from cognee.infrastructure.llm.exceptions import LLMQuotaExceededError


_NON_RETRYABLE_LLM_ERRORS = (
    litellm.exceptions.NotFoundError,
    litellm.exceptions.AuthenticationError,
    asyncio.CancelledError,
    LLMQuotaExceededError,
)

_QUOTA_EXHAUSTED_MARKERS = (
    "insufficient_quota",
    "quota_exceeded",
    "quota exceeded",
    "quota has been exceeded",
    "exceeded your current quota",
    "exceeded your quota",
    "billing hard limit",
    "billing limit",
    "billing quota",
    "payment required",
    "credit balance",
    "credits exhausted",
    "free quota",
)

_ERROR_TEXT_ATTRS = (
    "message",
    "body",
    "error",
    "code",
    "type",
    "status_code",
    "litellm_debug_info",
)


def _iter_exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = error

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _iter_response_text(response: Any) -> Iterator[str]:
    if response is None:
        return

    text = getattr(response, "text", None)
    if text:
        yield str(text)

    content = getattr(response, "content", None)
    if content:
        yield str(content)

    json_body = None
    try:
        json_body = response.json()
    except Exception:
        json_body = None

    if json_body is not None:
        yield repr(json_body)


def _iter_error_text(error: BaseException) -> Iterator[str]:
    for current in _iter_exception_chain(error):
        yield current.__class__.__name__
        yield str(current)

        for attr_name in _ERROR_TEXT_ATTRS:
            attr = getattr(current, attr_name, None)
            if attr:
                yield repr(attr)

        yield from _iter_response_text(getattr(current, "response", None))

        attr_dict = getattr(current, "__dict__", {})
        yield from _iter_response_text(attr_dict.get("response"))

        for attr_name in _ERROR_TEXT_ATTRS:
            attr = attr_dict.get(attr_name)
            if attr:
                yield repr(attr)


def is_quota_exceeded_error(error: BaseException) -> bool:
    error_text = "\n".join(_iter_error_text(error)).lower()
    return any(marker in error_text for marker in _QUOTA_EXHAUSTED_MARKERS)


def is_non_retryable_llm_error(error: BaseException) -> bool:
    return isinstance(error, _NON_RETRYABLE_LLM_ERRORS) or is_quota_exceeded_error(error)


def should_retry_llm_error(error: BaseException) -> bool:
    if isinstance(error, LLMQuotaExceededError):
        return False

    if is_quota_exceeded_error(error):
        raise LLMQuotaExceededError() from error

    return not isinstance(error, _NON_RETRYABLE_LLM_ERRORS)


retry_if_retryable_llm_error = retry_if_exception(should_retry_llm_error)
