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


class ProviderNotDeducibleError(CogneeValidationError):
    """
    Raised when ``llm_provider`` is not set and cannot be inferred from the
    ``llm_model`` prefix because the prefix is not a provider cognee supports.

    Tells the user to set the provider explicitly, since it could not be inferred.
    """

    def __init__(self, model: str) -> None:
        message = (
            f"Could not infer an LLM provider from LLM_MODEL={model!r}: the prefix "
            "is not a provider cognee supports. Set LLM_PROVIDER explicitly."
        )
        super().__init__(message=message, name="ProviderNotDeducibleError")


class MissingSystemPromptPathError(CogneeValidationError):
    def __init__(
        self,
        name: str = "MissingSystemPromptPathError",
    ) -> None:
        message = "No system prompt path provided."
        super().__init__(message, name)
