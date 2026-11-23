"""Shared tenacity retry policy for embedding calls.

The mirror of ``cognee.infrastructure.llm.retry_config`` for the embedding side:
each engine keeps its own terminal error classes at the decorator, while the one
rule every engine needs -- a spend cap cannot clear by retrying -- lives here.

Budget exhaustion cannot be recognised by exception class:

* ``litellm.BudgetExceededError`` subclasses plain ``Exception``, so no base
  class in a type tuple can match it;
* against a LiteLLM *proxy* -- the deployment where budgets are configured at
  all -- the client never receives that class. The proxy raises it server-side
  and the client-side litellm maps the status to an ordinary ``RateLimitError``;
* the engines re-raise provider failures as ``EmbeddingException(...) from
  error``, which hides the provider class from ``retry_if_not_exception_type``
  even when the class would have matched.

``is_budget_exhausted_error`` classifies on the signals that survive all three
(a 402 status, litellm's own class, the proxy's ``budget_exceeded`` body, and
the provider's budget sentence) and walks the ``__cause__`` chain, so it is
applied as a predicate rather than as a type tuple.

A missing provider SDK is classified the same way. litellm does not raise the
``ImportError`` itself: ``boto3`` missing for bedrock / sagemaker, or
``google-auth`` for vertex_ai, surfaces as an ``APIConnectionError`` (status 500)
raised *while handling* the ``ImportError``, so the ``ImportError`` is that link's
``__context__``, not its ``__cause__``. The engine then wraps the connection error
in ``EmbeddingException``. ``is_missing_package_error`` walks ``__cause__`` and
looks one ``__context__`` step down at each link, which covers exactly that shape.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from tenacity import RetryCallState, retry_if_exception

from cognee.infrastructure.llm.exceptions import (
    LLMPaymentRequiredError,
    is_budget_exhausted_error,
)


def is_missing_package_error(error: BaseException) -> bool:
    """Whether *error* failed on a package that is not installed.

    True for an ``ImportError`` anywhere down the ``__cause__`` chain, or one that a
    link was raised while handling (its direct ``__context__``): litellm's
    ``APIConnectionError`` for a missing provider SDK has that shape. Deeper
    ``__context__`` links are not followed, so an unrelated error raised while
    handling a stray ``ImportError`` is not misclassified.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ImportError) or isinstance(current.__context__, ImportError):
            return True
        current = current.__cause__
    return False


def embedding_retry_condition(*terminal_types: type[BaseException]) -> retry_if_exception:
    """Retry transient failures, but never *terminal_types*, budget exhaustion or a missing package.

    ``LLMPaymentRequiredError`` is terminal for every engine, so an engine that
    converts a budget rejection into the actionable 402 itself does not then run
    the backoff ladder on its own exception.
    """
    non_retryable: tuple[type[BaseException], ...] = (LLMPaymentRequiredError, *terminal_types)

    def should_retry(error: BaseException) -> bool:
        if isinstance(error, non_retryable):
            return False
        return not (is_budget_exhausted_error(error) or is_missing_package_error(error))

    return retry_if_exception(should_retry)


def wait_for_retry_after(fallback: Callable[[RetryCallState], float]):
    """Wait as long as the failure's ``Retry-After`` says, else use ``fallback``."""

    def wait(retry_state: RetryCallState) -> float:
        advertised = _advertised_wait(retry_state)
        return fallback(retry_state) if advertised is None else advertised

    return wait


def stop_when_retry_after_exceeds(window_seconds: float):
    """Stop instead of sleeping past ``window_seconds`` when ``Retry-After`` asks for more."""

    def stop(retry_state: RetryCallState) -> bool:
        advertised = _advertised_wait(retry_state)
        if advertised is None:
            return False
        return retry_state.seconds_since_start + advertised > window_seconds

    return stop


def _advertised_wait(retry_state: RetryCallState) -> float | None:
    outcome = retry_state.outcome
    if outcome is None or not outcome.failed:
        return None
    return retry_after_seconds(outcome.exception())


def retry_after_seconds(error: BaseException | None) -> float | None:
    """Seconds the provider asked to wait, from the first ``Retry-After`` in ``error``'s chain.

    Engines wrap provider errors (``EmbeddingException(...) from error``), so the
    cause/context chain is walked, as overload detection does. A link carries the
    value as a ``retry_after`` attribute or as a header of its ``response``.
    """
    seen: set[int] = set()
    node = error
    while node is not None and id(node) not in seen and len(seen) < 20:
        raw = getattr(node, "retry_after", None)
        if raw is None:
            headers = getattr(getattr(node, "response", None), "headers", None)
            raw = headers.get("retry-after") if headers is not None else None
        seconds = _parse_retry_after(raw)
        if seconds is not None:
            return seconds
        seen.add(id(node))
        node = node.__cause__ or node.__context__
    return None


def _parse_retry_after(raw) -> float | None:
    """Delta-seconds or an HTTP date as seconds from now; None when absent or unreadable."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw >= 0 else None
    if not isinstance(raw, str):
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            return None
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    return seconds if seconds >= 0 else None
