"""Get the LLM client."""

from enum import Enum

from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.ollama.adapter import (
    OllamaAPIAdapter,
)
from cognee.infrastructure.llm.exceptions import (
    LLMAPIKeyNotSetError,
    UnsupportedLLMProviderError,
)


# Define an Enum for LLM Providers
class LLMProvider(Enum):
    """
    Define an Enum for identifying different LLM Providers.

    This Enum includes the following members:
    - OPENAI: Represents the OpenAI provider.
    - OLLAMA: Represents the Ollama provider.
    - ANTHROPIC: Represents the Anthropic provider.
    - CUSTOM: Represents a custom provider option.
    - GEMINI: Represents the Gemini provider.
    - MISTRAL: Represents the Mistral AI provider.
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"
    GEMINI = "gemini"
    MISTRAL = "mistral"


def get_llm_client(raise_api_key_error: bool = True):
    """
    Get the LLM client based on the configuration using Enums.

    This function retrieves the configuration for the LLM provider and model, and
    initializes the appropriate LLM client adapter accordingly. It raises an
    LLMAPIKeyNotSetError if the LLM API key is not set for certain providers or if the provider
    is unsupported.

    Returns:
    --------

        An instance of the appropriate LLM client adapter based on the provider
        configuration.
    """
    llm_config = get_llm_config()

    provider = LLMProvider(llm_config.llm_provider)

    # Check if max_token value is defined in liteLLM for given model
    # if not use value from cognee configuration
    from cognee.infrastructure.llm.utils import (
        get_model_max_completion_tokens,
    )  # imported here to avoid circular imports

    model_max_completion_tokens = get_model_max_completion_tokens(llm_config.llm_model)
    max_completion_tokens = (
        model_max_completion_tokens
        if model_max_completion_tokens
        else llm_config.llm_max_completion_tokens
    )

    if provider == LLMProvider.OPENAI:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
            OpenAIAdapter,
        )

        return OpenAIAdapter(
            api_key=llm_config.llm_api_key,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            model=llm_config.llm_model,
            transcription_model=llm_config.transcription_model,
            max_completion_tokens=max_completion_tokens,
            streaming=llm_config.llm_streaming,
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=llm_config.fallback_endpoint,
            fallback_model=llm_config.fallback_model,
        )

    elif provider == LLMProvider.OLLAMA:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
            GenericAPIAdapter,
        )

        return OllamaAPIAdapter(
            llm_config.llm_endpoint,
            llm_config.llm_api_key,
            llm_config.llm_model,
            "Ollama",
            max_completion_tokens=max_completion_tokens,
        )

    elif provider == LLMProvider.ANTHROPIC:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter import (
            AnthropicAdapter,
        )

        return AnthropicAdapter(
            max_completion_tokens=max_completion_tokens, model=llm_config.llm_model
        )

    elif provider == LLMProvider.CUSTOM:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
            GenericAPIAdapter,
        )

        return GenericAPIAdapter(
            llm_config.llm_endpoint,
            llm_config.llm_api_key,
            llm_config.llm_model,
            "Custom",
            max_completion_tokens=max_completion_tokens,
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=llm_config.fallback_endpoint,
            fallback_model=llm_config.fallback_model,
        )

    elif provider == LLMProvider.GEMINI:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.gemini.adapter import (
            GeminiAdapter,
        )

        return GeminiAdapter(
            api_key=llm_config.llm_api_key,
            model=llm_config.llm_model,
            max_completion_tokens=max_completion_tokens,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
        )

    elif provider == LLMProvider.MISTRAL:
        if llm_config.llm_api_key is None:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral.adapter import (
            MistralAdapter,
        )

        return MistralAdapter(
            api_key=llm_config.llm_api_key,
            model=llm_config.llm_model,
            max_completion_tokens=max_completion_tokens,
            endpoint=llm_config.llm_endpoint,
        )

    elif provider == LLMProvider.MISTRAL:
        if llm_config.llm_api_key is None:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral.adapter import (
            MistralAdapter,
        )

        return MistralAdapter(
            api_key=llm_config.llm_api_key,
            model=llm_config.llm_model,
            max_completion_tokens=max_completion_tokens,
            endpoint=llm_config.llm_endpoint,
        )

    else:
        raise UnsupportedLLMProviderError(provider)
