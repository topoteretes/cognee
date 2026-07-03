import os

from cognee.infrastructure.databases.vector.embeddings.config import (
    get_embedding_context_config,
    _resolve_embedding_dimensions,
)
from cognee.infrastructure.llm.config import (
    get_llm_context_config,
)
from .derive_embedding_settings import derive_embedding_settings
from .EmbeddingEngine import EmbeddingEngine
from functools import lru_cache

# Any of these counts as "the user configured embeddings themselves" — a
# user setting only EMBEDDING_API_KEY relies on the documented OpenAI
# defaults for provider/model, and derivation must not override that.
_EMBEDDING_ENV_VARS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_ENDPOINT",
)


def _embeddings_explicitly_configured(config) -> bool:
    """True when the user chose an embedding provider/model themselves.

    Pydantic defaults make the config values useless for "was it set"
    detection, so check the process environment directly, plus
    ``model_fields_set`` — which catches .env-file values, programmatic
    ``cognee.config.set_embedding_*()`` calls and explicit context configs.
    """
    if any(os.environ.get(var) for var in _EMBEDDING_ENV_VARS):
        return True
    explicit_fields = {
        "embedding_provider",
        "embedding_model",
        "embedding_api_key",
        "embedding_endpoint",
    }
    return bool(explicit_fields & config.model_fields_set)


def _mock_embeddings_enabled() -> bool:
    return os.getenv("MOCK_EMBEDDING", "false").lower() in ("true", "1", "yes")


def get_embedding_engine() -> EmbeddingEngine:
    """
    Retrieve the embedding engine singleton based on configuration.

    This function calls the configuration retrieval functions to get the necessary settings
    for the embedding engine and creates a singleton instance. This ensures that too many
    requests won't be sent to HuggingFace by reusing the same instance for subsequent calls.

    When the user has not configured embeddings explicitly, the embedding
    provider is derived from the LLM provider instead of silently falling back
    to OpenAI with the wrong API key (see ``derive_embedding_settings``).

    Returns:
    --------

        - EmbeddingEngine: An instance of the embedding engine configured based on the
          retrieved settings.
    """
    config = get_embedding_context_config()
    llm_config = get_llm_context_config()

    if not _embeddings_explicitly_configured(config) and not _mock_embeddings_enabled():
        derived = derive_embedding_settings(
            llm_config.llm_provider,
            llm_config.llm_endpoint or None,
            llm_config.llm_api_key,
        )
        if derived is not None:
            dimensions = (
                derived["dimensions"]
                or _resolve_embedding_dimensions(derived["provider"], derived["model"])
                or config.embedding_dimensions
            )
            return create_embedding_engine(
                derived["provider"],
                derived["model"],
                dimensions,
                config.embedding_max_completion_tokens,
                derived["endpoint"],
                derived["api_key"],
                config.embedding_api_version,
                config.embedding_batch_size,
                config.huggingface_tokenizer or derived["huggingface_tokenizer"],
                llm_config.llm_api_key,
                llm_config.llm_provider,
            )

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
):
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

    Returns:
    --------

        Returns an instance of an embedding engine based on the specified provider.
    """
    if embedding_provider == "fastembed":
        from .FastembedEmbeddingEngine import FastembedEmbeddingEngine

        return FastembedEmbeddingEngine(
            model=embedding_model,
            dimensions=embedding_dimensions,
            max_completion_tokens=embedding_max_completion_tokens,
            batch_size=embedding_batch_size,
        )

    if embedding_provider == "ollama":
        from .OllamaEmbeddingEngine import OllamaEmbeddingEngine

        return OllamaEmbeddingEngine(
            model=embedding_model,
            dimensions=embedding_dimensions,
            max_completion_tokens=embedding_max_completion_tokens,
            endpoint=embedding_endpoint,
            huggingface_tokenizer=huggingface_tokenizer,
            batch_size=embedding_batch_size,
        )

    if embedding_provider == "openai_compatible":
        from .OpenAICompatibleEmbeddingEngine import OpenAICompatibleEmbeddingEngine

        return OpenAICompatibleEmbeddingEngine(
            model=embedding_model,
            dimensions=embedding_dimensions,
            max_completion_tokens=embedding_max_completion_tokens,
            endpoint=embedding_endpoint,
            api_key=embedding_api_key or llm_api_key,
            batch_size=embedding_batch_size,
        )

    from .LiteLLMEmbeddingEngine import LiteLLMEmbeddingEngine

    return LiteLLMEmbeddingEngine(
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
