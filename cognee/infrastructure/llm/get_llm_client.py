"""Get the LLM client."""
from enum import Enum
import json
import logging
# from cognee.infrastructure.llm import llm_config

from cognee.config import Config
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
    # logging.error(json.dumps(llm_config.to_dict()))
    provider = LLMProvider(config.llm_provider)

    if provider == LLMProvider.OPENAI:
        from .openai.adapter import OpenAIAdapter
        return OpenAIAdapter(llm_config.llm_api_key, llm_config.llm_model)
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
