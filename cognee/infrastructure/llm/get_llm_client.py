"""Get the LLM client."""

from enum import Enum

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.ollama.adapter import OllamaAPIAdapter


# Define an Enum for LLM Providers
class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"
    GEMINI = "gemini"


def get_llm_client():
    """Get the LLM client based on the configuration using Enums."""
    llm_config = get_llm_config()

    provider = LLMProvider(llm_config.llm_provider)

    # Check if max_token value is defined in liteLLM for given model
    # if not use value from cognee configuration
    from cognee.infrastructure.llm.utils import (
        get_model_max_tokens,
    )  # imported here to avoid circular imports

    model_max_tokens = get_model_max_tokens(llm_config.llm_model)
    max_tokens = model_max_tokens if model_max_tokens else llm_config.llm_max_tokens

    if provider == LLMProvider.OPENAI:
        if llm_config.llm_api_key is None:
            raise InvalidValueError(message="LLM API key is not set.")

        from .openai.adapter import OpenAIAdapter

        return OpenAIAdapter(
            api_key=llm_config.llm_api_key,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            model=llm_config.llm_model,
            transcription_model=llm_config.transcription_model,
            max_tokens=max_tokens,
            streaming=llm_config.llm_streaming,
        )

    elif provider == LLMProvider.OLLAMA:
        if llm_config.llm_api_key is None:
            raise InvalidValueError(message="LLM API key is not set.")

        from .generic_llm_api.adapter import GenericAPIAdapter

        return OllamaAPIAdapter(
            llm_config.llm_endpoint,
            llm_config.llm_api_key,
            llm_config.llm_model,
            "Ollama",
            max_tokens=max_tokens,
        )

    elif provider == LLMProvider.ANTHROPIC:
        from .anthropic.adapter import AnthropicAdapter

        return AnthropicAdapter(max_tokens=max_tokens, model=llm_config.llm_model)

    elif provider == LLMProvider.CUSTOM:
        if llm_config.llm_api_key is None:
            raise InvalidValueError(message="LLM API key is not set.")

        from .generic_llm_api.adapter import GenericAPIAdapter

        return GenericAPIAdapter(
            llm_config.llm_endpoint,
            llm_config.llm_api_key,
            llm_config.llm_model,
            "Custom",
            max_tokens=max_tokens,
        )

    elif provider == LLMProvider.GEMINI:
        if llm_config.llm_api_key is None:
            raise InvalidValueError(message="LLM API key is not set.")

        from .gemini.adapter import GeminiAdapter

        return GeminiAdapter(
            api_key=llm_config.llm_api_key,
            model=llm_config.llm_model,
            max_tokens=max_tokens,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            streaming=llm_config.llm_streaming,
        )

    else:
        raise InvalidValueError(message=f"Unsupported LLM provider: {provider}")
