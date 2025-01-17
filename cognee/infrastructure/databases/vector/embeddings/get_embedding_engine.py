from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.llm.config import get_llm_config
from .EmbeddingEngine import EmbeddingEngine
from .LiteLLMEmbeddingEngine import LiteLLMEmbeddingEngine


def get_embedding_engine() -> EmbeddingEngine:
    config = get_embedding_config()
    llm_config = get_llm_config()

    return LiteLLMEmbeddingEngine(
        # If OpenAI API is used for embeddings, litellm needs only the api_key.
        api_key=config.embedding_api_key or llm_config.llm_api_key,
        endpoint=config.embedding_endpoint,
        api_version=config.embedding_api_version,
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )
