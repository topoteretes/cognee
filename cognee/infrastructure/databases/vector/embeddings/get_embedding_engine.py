from functools import lru_cache

from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.vector.embeddings.config import (
    get_embedding_context_config,
    keyless_embedding_defaults_apply,
    resolve_embedding_defaults,
)
from cognee.infrastructure.llm.config import (
    get_llm_context_config,
)
from cognee.shared.logging_utils import get_logger

from .EmbeddingEngine import EmbeddingEngine
from .encoding import base64_supported, validate_encoding_format

logger = get_logger("get_embedding_engine")


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
    provider, model, dimensions = resolve_embedding_defaults(config, llm_config)
    encoding_format = validate_encoding_format(config.embedding_encoding_format)
    if encoding_format == "base64" and keyless_embedding_defaults_apply(config, llm_config):
        # With no key configured, embeddings run on local fastembed, which has no wire
        # format. The setting is for the provider a key will select, so it waits for it.
        _warn_base64_waits_for_a_key()
        encoding_format = None
    # Embedding engine has to be a singleton based on configuration to ensure too many requests won't be sent to HuggingFace
    return create_embedding_engine(
        provider,
        model,
        dimensions,
        config.embedding_max_completion_tokens,
        config.embedding_endpoint,
        config.embedding_api_key,
        config.embedding_api_version,
        config.embedding_batch_size,
        config.huggingface_tokenizer,
        llm_config.llm_api_key,
        llm_config.llm_provider,
        config.embedding_input_type,
        encoding_format,
    )


@lru_cache(maxsize=1)
def _warn_base64_waits_for_a_key() -> None:
    """Say once per process that EMBEDDING_ENCODING_FORMAT=base64 is not applied."""
    logger.warning(
        "EMBEDDING_ENCODING_FORMAT=base64 is not applied: no LLM or embedding key is "
        "configured, so embeddings run locally on fastembed, which has no wire format. "
        "It applies once a key selects a remote embedding provider."
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
    embedding_input_type=None,
    embedding_encoding_format=None,
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
    embedding_encoding_format = validate_encoding_format(embedding_encoding_format)
    if embedding_encoding_format == "base64" and not base64_supported(embedding_provider):
        raise CogneeConfigurationError(
            f"The embedding provider '{embedding_provider}' has no wire format to request, "
            f"so EMBEDDING_ENCODING_FORMAT=base64 cannot apply to it.",
            remediation=(
                "Comment out or remove EMBEDDING_ENCODING_FORMAT in your .env file or "
                "environment. Embeddings are unchanged; base64 only makes responses cheaper "
                "to parse."
            ),
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
            api_key=embedding_api_key or llm_api_key,
            batch_size=embedding_batch_size,
            input_type=embedding_input_type,
            encoding_format=embedding_encoding_format,
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
        input_type=embedding_input_type,
        encoding_format=embedding_encoding_format,
    )
