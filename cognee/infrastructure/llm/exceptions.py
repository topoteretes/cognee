from cognee.exceptions import CogneeConfigurationError, CogneeValidationError


class ContentPolicyFilterError(CogneeValidationError):
    pass


class LLMAPIKeyNotSetError(CogneeConfigurationError):
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
