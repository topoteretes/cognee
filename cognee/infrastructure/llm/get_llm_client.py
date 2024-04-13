"""Get the LLM client."""
from enum import Enum
from cognee.config import Config
from .anthropic.adapter import AnthropicAdapter
from .openai.adapter import OpenAIAdapter
from .generic_llm_api.adapter import GenericAPIAdapter

# Define an Enum for LLM Providers
class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"

config = Config()
config.load()

def get_llm_client():
    """Get the LLM client based on the configuration using Enums."""
    provider = LLMProvider(config.llm_provider)

    if provider == LLMProvider.OPENAI:
        return OpenAIAdapter(config.openai_key, config.openai_model)
    elif provider == LLMProvider.OLLAMA:
        return GenericAPIAdapter(config.ollama_endpoint, config.ollama_key, config.ollama_model)
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicAdapter(config.custom_model)
    elif provider == LLMProvider.CUSTOM:
        return GenericAPIAdapter(config.custom_endpoint, config.custom_key, config.custom_model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
