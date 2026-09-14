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
"""

from tenacity import retry_if_exception

from cognee.infrastructure.llm.exceptions import (
    LLMPaymentRequiredError,
    is_budget_exhausted_error,
)


def embedding_retry_condition(*terminal_types: type[BaseException]) -> retry_if_exception:
    """Retry transient failures, but never *terminal_types* or budget exhaustion.

    ``LLMPaymentRequiredError`` is terminal for every engine, so an engine that
    converts a budget rejection into the actionable 402 itself does not then run
    the backoff ladder on its own exception.
    """
    non_retryable: tuple[type[BaseException], ...] = (LLMPaymentRequiredError, *terminal_types)

    def should_retry(error: BaseException) -> bool:
        if isinstance(error, non_retryable):
            return False
        return not is_budget_exhausted_error(error)

    return retry_if_exception(should_retry)
