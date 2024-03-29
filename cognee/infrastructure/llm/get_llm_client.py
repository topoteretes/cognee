"""Get the LLM client."""
from enum import Enum
from cognee.config import Config
from .anthropic.adapter import AnthropicAdapter
from .openai.adapter import OpenAIAdapter
from .generic_llm_api.adapter import GenericAPIAdapter
import logging
logging.basicConfig(level=logging.INFO)

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
        print("Using OpenAI API")
        return OpenAIAdapter(config.openai_key, config.model)
    elif provider == LLMProvider.OLLAMA:
        print("Using Ollama API")
        return GenericAPIAdapter(config.ollama_endpoint, config.ollama_key, config.ollama_model)
    elif provider == LLMProvider.ANTHROPIC:
        print("Using Anthropic API")
        return AnthropicAdapter(config.custom_endpoint, config.custom_endpoint, config.custom_model)
    elif provider == LLMProvider.CUSTOM:
        print("Using Custom API")
        return GenericAPIAdapter(config.custom_endpoint, config.custom_key, config.custom_model)
        # Add your custom LLM provider here
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# Usage example
llm_client = get_llm_client()
