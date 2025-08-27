from cognee.exceptions.exceptions import CogneeValidationError


class ContentPolicyFilterError(CogneeValidationError):
    pass


class LLMAPIKeyNotSetError(CogneeValidationError):
    """
    Raised when the LLM API key is not set in the configuration.
    """

    def __init__(self, message: str = "LLM API key is not set."):
        super().__init__(message=message, name="LLMAPIKeyNotSetError")


class UnsupportedLLMProviderError(CogneeValidationError):
    """
    Raised when an unsupported LLM provider is specified in the configuration.
    """

    def __init__(self, provider: str):
        message = f"Unsupported LLM provider: {provider}"
        super().__init__(message=message, name="UnsupportedLLMProviderError")


class MissingSystemPromptPathError(CogneeValidationError):
    def __init__(
        self,
        name: str = "MissingSystemPromptPathError",
    ):
        message = "No system prompt path provided."
        super().__init__(message, name)
