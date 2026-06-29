from fastapi import status

from cognee.exceptions.exceptions import CogneeValidationError


class ContentPolicyFilterError(CogneeValidationError):
    pass


class LLMAPIKeyNotSetError(CogneeValidationError):
    """
    Raised when the LLM API key is not set in the configuration.
    """

    def __init__(self, message: str = "LLM API key is not set.") -> None:
        super().__init__(message=message, name="LLMAPIKeyNotSetError")


class LLMQuotaExceededError(CogneeValidationError):
    """
    Raised when the configured LLM provider reports exhausted quota or billing.
    """

    def __init__(
        self,
        message: str = (
            "LLM provider quota or billing limit was exhausted. "
            "Check provider credits, billing limits, or use a different model/API key before retrying."
        ),
    ) -> None:
        super().__init__(
            message=message,
            name="LLMQuotaExceededError",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )


class UnsupportedLLMProviderError(CogneeValidationError):
    """
    Raised when an unsupported LLM provider is specified in the configuration.
    """

    def __init__(self, provider: str) -> None:
        message = f"Unsupported LLM provider: {provider}"
        super().__init__(message=message, name="UnsupportedLLMProviderError")


class MissingSystemPromptPathError(CogneeValidationError):
    def __init__(
        self,
        name: str = "MissingSystemPromptPathError",
    ) -> None:
        message = "No system prompt path provided."
        super().__init__(message, name)
