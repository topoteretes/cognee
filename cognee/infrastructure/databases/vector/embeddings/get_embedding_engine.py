from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.llm.config import (
    get_llm_config,
)
from .EmbeddingEngine import EmbeddingEngine
from functools import lru_cache


def get_embedding_engine() -> EmbeddingEngine:
    """
    Retrieve the embedding engine singleton based on configuration.

    This function calls the configuration retrieval functions to get the necessary settings
    for the embedding engine and creates a singleton instance. This ensures that too many
    requests won't be sent to HuggingFace by reusing the same instance for subsequent calls.

    Returns:
    --------

        - EmbeddingEngine: An instance of the embedding engine configured based on the
          retrieved settings.
    """
    config = get_embedding_config()
    llm_config = get_llm_config()
    # Embedding engine has to be a singleton based on configuration to ensure too many requests won't be sent to HuggingFace
    return create_embedding_engine(
        config.embedding_provider,
        config.embedding_model,
        config.embedding_dimensions,
        config.embedding_max_completion_tokens,
        config.embedding_endpoint,
        config.embedding_api_key,
        config.embedding_api_version,
        config.embedding_batch_size,
        config.huggingface_tokenizer,
        llm_config.llm_api_key,
        llm_config.llm_provider,
        config.accumulate_embedding_calls,
        config.accumulate_embedding_timeout_ms,
    )


@lru_cache
def create_embedding_engine(
    embedding_provider,
    embedding_model,
    embedding_dimensions,
    embedding_max_completion_tokens,
    embedding_endpoint,
    embedding_api_key,
    embedding_api_version,
    embedding_batch_size,
    huggingface_tokenizer,
    llm_api_key,
    llm_provider,
    accumulate_embedding_calls: bool = True,
    accumulate_embedding_timeout_ms: int = 20,
) -> EmbeddingEngine:
    """
    Create and return an embedding engine based on the specified provider.

    Parameters:
    -----------

        - embedding_provider: The name of the embedding provider, e.g., 'fastembed',
          'ollama', or another supported provider.
        - embedding_model: The model to be used for the embedding engine.
        - embedding_dimensions: The number of dimensions for the embeddings.
        - embedding_max_completion_tokens: The maximum number of tokens for the embeddings.
        - embedding_endpoint: The endpoint for the embedding service, relevant for certain
          providers.
        - embedding_api_key: API key to authenticate with the embedding service, if
          required.
        - embedding_api_version: Version of the API to be used for the embedding service, if
          applicable.
        - huggingface_tokenizer: Tokenizer from Hugging Face for tokenizing input text, used
          for specific providers.
        - llm_api_key: API key for the LLM service, to be used if embedding_api_key is not
          provided.
        - accumulate_embedding_calls: Whether to enable coalescing of concurrent embed_text
          calls into batched API requests (default: True).
        - accumulate_embedding_timeout_ms: Maximum wait time in milliseconds before flushing
          accumulated calls (default: 20).

    Returns:
    --------

        Returns an instance of an embedding engine based on the specified provider.
    """
    if embedding_provider == "fastembed":
        from .FastembedEmbeddingEngine import FastembedEmbeddingEngine

        engine = FastembedEmbeddingEngine(
            model=embedding_model,
            dimensions=embedding_dimensions,
            max_completion_tokens=embedding_max_completion_tokens,
            batch_size=embedding_batch_size,
        )
    elif embedding_provider == "ollama":
        from .OllamaEmbeddingEngine import OllamaEmbeddingEngine

        engine = OllamaEmbeddingEngine(
            model=embedding_model,
            dimensions=embedding_dimensions,
            max_completion_tokens=embedding_max_completion_tokens,
            endpoint=embedding_endpoint,
            huggingface_tokenizer=huggingface_tokenizer,
            batch_size=embedding_batch_size,
        )
    elif embedding_provider == "openai_compatible":
        from .OpenAICompatibleEmbeddingEngine import OpenAICompatibleEmbeddingEngine

        engine = OpenAICompatibleEmbeddingEngine(
            model=embedding_model,
            dimensions=embedding_dimensions,
            max_completion_tokens=embedding_max_completion_tokens,
            endpoint=embedding_endpoint,
            api_key=embedding_api_key or llm_api_key,
            batch_size=embedding_batch_size,
        )
    else:
        from .LiteLLMEmbeddingEngine import LiteLLMEmbeddingEngine

        engine = LiteLLMEmbeddingEngine(
            provider=embedding_provider,
            api_key=embedding_api_key
            or (embedding_api_key if llm_provider == "custom" else llm_api_key),
            endpoint=embedding_endpoint,
            api_version=embedding_api_version,
            model=embedding_model,
            dimensions=embedding_dimensions,
            max_completion_tokens=embedding_max_completion_tokens,
            batch_size=embedding_batch_size,
        )

    if accumulate_embedding_calls:
        from .AccumulatingEmbeddingEngine import AccumulatingEmbeddingEngine

        engine = AccumulatingEmbeddingEngine(
            inner=engine,
            flush_timeout_seconds=accumulate_embedding_timeout_ms / 1000.0,
        )

    return engine
