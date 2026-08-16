from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_context_config
from cognee.infrastructure.llm.config import (
    get_llm_context_config,
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
    config = get_embedding_context_config()
    llm_config = get_llm_context_config()
    embedding_configured_fields = frozenset(getattr(config, "model_fields_set", set()))
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
        embedding_configured_fields,
    )


def _normalize_provider(provider):
    return provider.lower() if isinstance(provider, str) else provider


def _infer_provider_from_model(model):
    if not isinstance(model, str) or "/" not in model:
        return None

    return model.split("/", 1)[0].lower()


def _embedding_identity_is_explicit(embedding_model, configured_fields):
    if configured_fields is None:
        configured_fields = set()

    if "embedding_provider" in configured_fields:
        return True

    if "embedding_model" in configured_fields and _infer_provider_from_model(embedding_model):
        return True

    return False


def _validate_embedding_engine_config(
    embedding_provider,
    embedding_model,
    embedding_dimensions,
    llm_provider,
    embedding_configured_fields,
):
    if not embedding_provider or not embedding_model:
        raise CogneeConfigurationError(
            "Embedding configuration is incomplete. Set EMBEDDING_PROVIDER and "
            "EMBEDDING_MODEL explicitly."
        )

    if embedding_dimensions is None:
        raise CogneeConfigurationError(
            "Embedding configuration is missing dimensions. Set EMBEDDING_DIMENSIONS "
            "explicitly for this embedding model."
        )

    normalized_llm_provider = _normalize_provider(llm_provider)
    normalized_embedding_provider = _normalize_provider(embedding_provider)
    explicit_embedding_identity = _embedding_identity_is_explicit(
        embedding_model, embedding_configured_fields
    )

    if (
        normalized_llm_provider
        and normalized_llm_provider != "openai"
        and normalized_embedding_provider == "openai"
        and not explicit_embedding_identity
    ):
        raise CogneeConfigurationError(
            "LLM_PROVIDER is set to a non-OpenAI provider, but embedding configuration "
            "is still using Cognee's default OpenAI embeddings. Set EMBEDDING_PROVIDER "
            "and EMBEDDING_MODEL explicitly, or set an explicit OpenAI embedding model "
            "and EMBEDDING_API_KEY if you intend to use OpenAI embeddings."
        )


def _get_embedding_api_key(embedding_api_key, llm_api_key, embedding_provider, llm_provider):
    if embedding_api_key:
        return embedding_api_key

    normalized_embedding_provider = _normalize_provider(embedding_provider)
    normalized_llm_provider = _normalize_provider(llm_provider)

    if normalized_embedding_provider == normalized_llm_provider:
        return llm_api_key

    if normalized_embedding_provider == "openai" and normalized_llm_provider in (None, "openai"):
        return llm_api_key

    return None


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
    embedding_configured_fields=frozenset(),
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
    _validate_embedding_engine_config(
        embedding_provider,
        embedding_model,
        embedding_dimensions,
        llm_provider,
        embedding_configured_fields,
    )
    api_key = _get_embedding_api_key(
        embedding_api_key, llm_api_key, embedding_provider, llm_provider
    )

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
            api_key=api_key,
            batch_size=embedding_batch_size,
        )

    from .LiteLLMEmbeddingEngine import LiteLLMEmbeddingEngine

    return LiteLLMEmbeddingEngine(
        provider=embedding_provider,
        api_key=api_key,
        endpoint=embedding_endpoint,
        api_version=embedding_api_version,
        model=embedding_model,
        dimensions=embedding_dimensions,
        max_completion_tokens=embedding_max_completion_tokens,
        batch_size=embedding_batch_size,
    )
