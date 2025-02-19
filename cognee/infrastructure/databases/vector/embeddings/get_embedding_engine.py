from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.llm.config import get_llm_config
from .EmbeddingEngine import EmbeddingEngine


def get_embedding_engine() -> EmbeddingEngine:
    config = get_embedding_config()
    llm_config = get_llm_config()

    if config.embedding_provider == "fastembed":
        from .FastembedEmbeddingEngine import FastembedEmbeddingEngine

        return FastembedEmbeddingEngine(
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            max_tokens=config.embedding_max_tokens,
        )

    if config.embedding_provider == "ollama":
        from .OllamaEmbeddingEngine import OllamaEmbeddingEngine

        return OllamaEmbeddingEngine(
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            max_tokens=config.embedding_max_tokens,
            huggingface_tokenizer=config.huggingface_tokenizer,
        )

    from .LiteLLMEmbeddingEngine import LiteLLMEmbeddingEngine

    return LiteLLMEmbeddingEngine(
        provider=config.embedding_provider,
        api_key=config.embedding_api_key or llm_config.llm_api_key,
        endpoint=config.embedding_endpoint,
        api_version=config.embedding_api_version,
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
        max_tokens=config.embedding_max_tokens,
    )
