"""Get the LLM client."""

from enum import Enum

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm import get_llm_config


# Define an Enum for LLM Providers
class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


def get_llm_client():
    """Get the LLM client based on the configuration using Enums."""
    llm_config = get_llm_config()

    provider = LLMProvider(llm_config.llm_provider)

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
            streaming=llm_config.llm_streaming,
        )

    elif provider == LLMProvider.OLLAMA:
        if llm_config.llm_api_key is None:
            raise InvalidValueError(message="LLM API key is not set.")

        from .generic_llm_api.adapter import GenericAPIAdapter

        return GenericAPIAdapter(
            llm_config.llm_endpoint, llm_config.llm_api_key, llm_config.llm_model, "Ollama"
        )

    elif provider == LLMProvider.ANTHROPIC:
        from .anthropic.adapter import AnthropicAdapter

        return AnthropicAdapter(llm_config.llm_model)

    elif provider == LLMProvider.CUSTOM:
        if llm_config.llm_api_key is None:
            raise InvalidValueError(message="LLM API key is not set.")

        from .generic_llm_api.adapter import GenericAPIAdapter

        return GenericAPIAdapter(
            llm_config.llm_endpoint, llm_config.llm_api_key, llm_config.llm_model, "Custom"
        )

    else:
        raise InvalidValueError(message=f"Unsupported LLM provider: {provider}")
