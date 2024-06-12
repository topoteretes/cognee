from cognee.infrastructure.llm import get_llm_config
from .EmbeddingEngine import EmbeddingEngine
from .LiteLLMEmbeddingEngine import LiteLLMEmbeddingEngine

def get_embedding_engine() -> EmbeddingEngine:
    llm_config = get_llm_config()
    return LiteLLMEmbeddingEngine(api_key = llm_config.llm_api_key)
