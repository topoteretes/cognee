import re

from cognee.exceptions.exceptions import CogneeValidationError


class ContentPolicyFilterError(CogneeValidationError):
    pass


class LLMPaymentRequiredError(CogneeValidationError):
    """Raised when the LLM provider returns HTTP 402 (payment required / budget exhausted)."""

    def __init__(
        self, message: str = "LLM provider requires payment or token budget is exhausted."
    ) -> None:
        super().__init__(message=message, name="LLMPaymentRequiredError", status_code=402)


# Message text is the only signal that reliably survives the wrapper exceptions
# the instructor adapters raise: by the time a LiteLLM-proxy budget rejection
# reaches us it is an ``InstructorRetryException`` whose deepest cause is a
# client-side ``RateLimitError`` — a different class from
# ``litellm.BudgetExceededError``, and one whose response body has already been
# consumed, so the structured ``error.type`` check below cannot fire.
#
# LiteLLM raises ``BudgetExceededError`` with three distinct sentence shapes,
# between them covering every budget scope:
#   1. "Budget has been exceeded! [Scope=...] Current cost: X, Max budget: Y"
#      -- virtual key, team member, team, project, organization, tag
#   2. "ExceededBudget: [End ]User=<id> over budget. Spend=X, Budget=Y"
#      -- internal-user and end-user budgets
#   3. "LiteLLM {Virtual Key|End User}: <id>, exceeded budget for model=<m>"
#      -- per-model budget caps
#
# One regex both detects and extracts, so a positive match always yields a
# detail. Matching the WHOLE sentence — rather than checking for fragments
# anywhere in the text — is what keeps this safe: ``str(InstructorRetryException)``
# concatenates the model's own partial completions, so loose fragment matching
# fires on any document that merely mentions an exceeded budget and a maximum
# somewhere in several KB of unrelated prose. The bounded ``.{0,200}?`` spans
# only the variable scope segment, keeps the scan linear, and caps the detail.
_BUDGET_SENTENCE_RE = re.compile(
    r"Budget has been exceeded!.{0,200}?Max budget:\s*[\d.]+"
    r"|ExceededBudget:.{0,200}?Budget=[\d.]+"
    r"|LiteLLM (?:Virtual Key|End User):.{0,200}?exceeded budget for model=\S{1,100}",
    re.IGNORECASE | re.DOTALL,
)

# The sentence can carry the virtual-key hash, key alias, and end-user / team /
# organization IDs, and the detail is returned to API callers in the 402 body.
# Spend and budget figures are kept — they are the caller's own, and are the
# actionable part — but identifiers are masked.
_BUDGET_IDENTIFIER_RE = re.compile(
    r"\b(Virtual Key|End User|User|Team|Project|Organization|Tag|key_alias)(\s*[:=]\s*)([^\s,]+)",
    re.IGNORECASE,
)


def _redact_budget_identifiers(sentence: str) -> str:
    return _BUDGET_IDENTIFIER_RE.sub(r"\1\2<redacted>", sentence)


def _has_budget_message(text: str) -> bool:
    return _BUDGET_SENTENCE_RE.search(text) is not None


def _is_budget_exhausted_link(e: BaseException) -> bool:
    """Classify a single exception in a ``__cause__`` chain."""
    # Case 1: provider-level payment required
    if getattr(e, "status_code", None) == 402:
        return True

    # Case 2: litellm library budget manager
    try:
        import litellm

        if isinstance(e, litellm.BudgetExceededError):
            return True
    except ImportError:
        pass

    # Case 3: LiteLLM proxy budget_exceeded encoded inside a 429. Only reachable
    # when the error is caught before the response body is consumed.
    if getattr(e, "status_code", None) == 429:
        response = getattr(e, "response", None)
        if response is not None:
            try:
                body = response.json()
                error = body.get("error", body)
                if isinstance(error, dict) and error.get("type") == "budget_exceeded":
                    return True
            except Exception:
                pass

    # Case 4: message text, the wrapper-proof fallback. ``str()`` is guarded
    # because this runs inside tenacity's retry predicate, where an exception
    # with a raising ``__str__`` would escape the retry machinery itself.
    try:
        text = str(e)
    except Exception:
        return False
    return _has_budget_message(text)


def is_budget_exhausted_error(e: BaseException) -> bool:
    """Return True if e signals LLM budget or payment exhaustion.

    Walks the ``__cause__`` chain, because adapters and instructor wrap the
    provider error with ``raise ... from``. ``__context__`` is deliberately not
    followed, so an unrelated error merely raised while handling a budget error
    is not misclassified.

    Four signals are checked per link: HTTP 402, ``litellm.BudgetExceededError``,
    the LiteLLM proxy's ``error.type == "budget_exceeded"`` body, and finally the
    message wording (see ``_BUDGET_SENTENCE_RE``).
    """
    seen: set[int] = set()
    current: BaseException | None = e
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_budget_exhausted_link(current):
            return True
        current = current.__cause__
    return False


def budget_exhaustion_detail(e: BaseException) -> str | None:
    """Pull the provider's own budget sentence out of a wrapped exception.

    ``str(e)`` on an ``InstructorRetryException`` is long and embeds the model's
    partial completions, so the sentence is extracted rather than passed whole,
    and identifiers within it are masked before it reaches an API response.

    Walks the ``__cause__`` chain like ``is_budget_exhausted_error``, since a
    positive classification may come from a link below the outermost exception.
    """
    seen: set[int] = set()
    current: BaseException | None = e
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        try:
            text = str(current)
        except Exception:
            text = ""
        match = _BUDGET_SENTENCE_RE.search(text)
        if match:
            return _redact_budget_identifiers(match.group(0).strip())
        current = current.__cause__
    return None


def raise_if_budget_exhausted(error: BaseException) -> None:
    """Re-raise budget exhaustion as ``LLMPaymentRequiredError`` (HTTP 402).

    Instructor reports provider failures as ``InstructorRetryException``, which
    the adapters catch in an earlier ``except`` clause than their budget
    handler. The check therefore has to happen at that re-raise site, or the
    budget handler further down is never reached.
    """
    if not is_budget_exhausted_error(error):
        return
    detail = budget_exhaustion_detail(error)
    if detail:
        raise LLMPaymentRequiredError(f"LLM budget exhausted: {detail}") from error
    raise LLMPaymentRequiredError() from error


class LLMAPIKeyNotSetError(CogneeValidationError):
    """
    Raised when the LLM API key is not set in the configuration.
    """

    def __init__(self, message: str = "LLM API key is not set.") -> None:
        super().__init__(message=message, name="LLMAPIKeyNotSetError")


class UnsupportedLLMProviderError(CogneeValidationError):
    """
    Raised when an unsupported LLM provider is specified in the configuration.
    """

    def __init__(self, provider: str) -> None:
        message = f"Unsupported LLM provider: {provider}"
        super().__init__(message=message, name="UnsupportedLLMProviderError")


class LLMQuotaExceededError(CogneeValidationError):
    """Raised when an LLM provider reports non-retryable quota or billing exhaustion."""

    def __init__(self, detail: str | None = None) -> None:
        message = (
            "LLM provider quota or billing limit was reached. This is not retryable. "
            "Check the provider billing/quota dashboard, raise the limit, or switch credentials."
        )
        if detail:
            message = f"{message} Provider error: {detail}"
        super().__init__(message=message, name="LLMQuotaExceededError")


class ProviderNotDeducibleError(CogneeValidationError):
    """
    Raised when ``llm_provider`` is not set and cannot be inferred from the
    ``llm_model`` prefix because the prefix is not a provider cognee supports.

    The message names the supported providers and points OpenAI-compatible or
    other litellm-routed prefixes (e.g. ``openrouter/``, ``groq/``, ``deepseek/``)
    at ``LLM_PROVIDER="custom"``, so the user is told exactly what to set.
    """

    def __init__(self, model: str) -> None:
        # Imported lazily: config.py is fully loaded by the time this is raised
        # (from its own validator), while importing it at module top is circular.
        from cognee.infrastructure.llm.config import KNOWN_LLM_PROVIDERS

        supported = ", ".join(sorted(KNOWN_LLM_PROVIDERS))
        message = (
            f"Could not infer an LLM provider from LLM_MODEL={model!r}: its prefix "
            f"is not a provider cognee supports. Set LLM_PROVIDER explicitly to one "
            f"of: {supported}. For OpenAI-compatible or other litellm-routed "
            f'endpoints (e.g. openrouter, groq, deepseek), use LLM_PROVIDER="custom".'
        )
        super().__init__(message=message, name="ProviderNotDeducibleError")


class MissingSystemPromptPathError(CogneeValidationError):
    def __init__(
        self,
        name: str = "MissingSystemPromptPathError",
    ) -> None:
        message = "No system prompt path provided."
        super().__init__(message, name)


class MCPSamplingUnavailableError(CogneeValidationError):
    """
    Raised when `LLM_PROVIDER=mcp-sampling` is selected but no host MCP sampling
    session is available (cognee is not running inside an MCP server, or the host
    did not grant the `sampling` capability).
    """

    def __init__(
        self,
        message: str = (
            "No MCP sampling session is available. LLM_PROVIDER=mcp-sampling only works while "
            "cognee runs as an MCP server inside a host that grants the `sampling` capability "
            "(support varies by host). Set LLM_PROVIDER to a provider with credentials, or "
            "run inside such a host."
        ),
    ) -> None:
        super().__init__(message=message, name="MCPSamplingUnavailableError")
