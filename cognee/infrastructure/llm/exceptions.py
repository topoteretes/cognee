from cognee.exceptions import CogneeConfigurationError, CogneeValidationError


class ContentPolicyFilterError(CogneeValidationError):
    pass


class LLMAPIKeyNotSetError(CogneeConfigurationError):
class LLMPaymentRequiredError(CogneeValidationError):
    """Raised when the LLM provider returns HTTP 402 (payment required / budget exhausted)."""

    def __init__(
        self, message: str = "LLM provider requires payment or token budget is exhausted."
    ) -> None:
        super().__init__(message=message, name="LLMPaymentRequiredError", status_code=402)


def is_budget_exhausted_error(e: Exception) -> bool:
    """Return True if e signals LLM budget or payment exhaustion.

    Three cases are handled:
    1. Any provider returning HTTP 402 Payment Required directly.
    2. litellm's own budget manager raising BudgetExceededError (status_code 429,
       rate_limit_type "budget") when litellm is used as a library with a configured budget.
    3. LiteLLM proxy (≥v1.x) returning HTTP 429 with a JSON body whose "error.type" field
       equals "budget_exceeded". This is LiteLLM-proxy-specific: the proxy enforces virtual-key
       and per-user spend caps and uses this response shape to distinguish budget exhaustion from
       ordinary rate limiting. If LiteLLM changes this response format this branch silently falls
       through, so callers should monitor for unhandled 429s after proxy upgrades.
    """
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

    # Case 3: LiteLLM proxy budget_exceeded encoded inside a 429
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

    return False


class LLMAPIKeyNotSetError(CogneeValidationError):
    """
    Raised when the LLM API key is not set in the configuration.
    """

    def __init__(self, message: str = "LLM API key is not set.") -> None:
        super().__init__(
            message=message,
            name="LLMAPIKeyNotSetError",
            remediation="Set LLM_API_KEY in your environment or .env file.",
        )


class UnsupportedLLMProviderError(CogneeConfigurationError):
    """
    Raised when an unsupported LLM provider is specified in the configuration.
    """

    def __init__(self, provider: str) -> None:
        message = f"Unsupported LLM provider: {provider}"
        super().__init__(
            message=message,
            name="UnsupportedLLMProviderError",
            remediation="Set LLM_PROVIDER to a supported provider.",
        )


class MissingSystemPromptPathError(CogneeValidationError):
    def __init__(
        self,
        name: str = "MissingSystemPromptPathError",
    ) -> None:
        message = "No system prompt path provided."
        super().__init__(message, name)
