"""Get the LLM client."""
from cognee.config import Config
from .openai.adapter import OpenAIAdapter

config = Config()
config.load()

def get_llm_client():
    """Get the LLM client."""
    return OpenAIAdapter(config.openai_key, config.model)