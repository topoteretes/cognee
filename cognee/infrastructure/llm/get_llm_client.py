"""Get the LLM client."""
from enum import Enum
from cognee.config import Config
from .openai.adapter import OpenAIAdapter
from .ollama.adapter import OllamaAPIAdapter

# Define an Enum for LLM Providers
class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

config = Config()
config.load()

def get_llm_client():
    """Get the LLM client based on the configuration using Enums."""
    provider = LLMProvider(config.llm_provider)

    if provider == LLMProvider.OPENAI:
        return OpenAIAdapter(config.openai_key, config.model)
    elif provider == LLMProvider.OLLAMA:
        return OllamaAPIAdapter(config.ollama_endpoint, config.ollama_key, config.ollama_model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# Usage example
llm_client = get_llm_client()
