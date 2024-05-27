"""Get the LLM client."""
from enum import Enum
from cognee.infrastructure.llm import get_llm_config

# Define an Enum for LLM Providers
class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"

llm_config = get_llm_config()
def get_llm_client():
    """Get the LLM client based on the configuration using Enums."""
    llm_config = get_llm_config()

    provider = LLMProvider(llm_config.llm_provider)

    if provider == LLMProvider.OPENAI:
        from .openai.adapter import OpenAIAdapter
        return OpenAIAdapter(llm_config.llm_api_key, llm_config.llm_model, llm_config.llm_streaming)
    elif provider == LLMProvider.OLLAMA:
        from .generic_llm_api.adapter import GenericAPIAdapter
        return GenericAPIAdapter(llm_config.llm_endpoint, llm_config.llm_api_key, llm_config.llm_model, "Ollama")
    elif provider == LLMProvider.ANTHROPIC:
        from .anthropic.adapter import AnthropicAdapter
        return AnthropicAdapter(llm_config.llm_model)
    elif provider == LLMProvider.CUSTOM:
        from .generic_llm_api.adapter import GenericAPIAdapter
        return GenericAPIAdapter(llm_config.llm_endpoint, llm_config.llm_api_key, llm_config.llm_model, "Custom")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
