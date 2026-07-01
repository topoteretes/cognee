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
            "(e.g. Claude Code / Cursor). Set LLM_PROVIDER to a provider with credentials, or "
            "run inside such a host."
        ),
    ) -> None:
        super().__init__(message=message, name="MCPSamplingUnavailableError")
