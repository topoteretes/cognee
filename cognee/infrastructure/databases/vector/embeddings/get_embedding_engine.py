from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.llm.config import get_llm_config
from .EmbeddingEngine import EmbeddingEngine
from .LiteLLMEmbeddingEngine import LiteLLMEmbeddingEngine


def get_embedding_engine() -> EmbeddingEngine:
    config = get_embedding_config()
    llm_config = get_llm_config()
    
    # Get provider-specific configurations
    provider = llm_config.llm_provider
    provider_config = LiteLLMEmbeddingEngine.PROVIDER_CONFIGS.get(provider, {})

    # Build engine arguments
    engine_args = {
        "provider": provider,
        "api_key": config.embedding_api_key or llm_config.llm_api_key,
    }

    # Add optional endpoint and api_version if they exist
    if config.embedding_endpoint:
        engine_args["endpoint"] = config.embedding_endpoint
    if config.embedding_api_version:
        engine_args["api_version"] = config.embedding_api_version

    # Use provider-specific model and dimensions if available,
    # otherwise fall back to config values
    engine_args["model"] = provider_config.get("model", config.embedding_model)
    engine_args["dimensions"] = provider_config.get("dimensions", config.embedding_dimensions)

    return LiteLLMEmbeddingEngine(**engine_args)