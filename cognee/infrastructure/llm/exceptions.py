from cognee.exceptions.exceptions import CogneeValidationError


class ContentPolicyFilterError(CogneeValidationError):
    pass


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


class MissingSystemPromptPathError(CogneeValidationError):
    def __init__(
        self,
        name: str = "MissingSystemPromptPathError",
    ) -> None:
        message = "No system prompt path provided."
        super().__init__(message, name)
