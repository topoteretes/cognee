from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices

from cognee.shared.logging_utils import get_logger


logger = get_logger("embedding_config")


# Hard fallback when neither litellm nor fastembed knows the model. This used
# to be the unconditional default and was the source of the silent
# "Vector(3072)" mismatch on every non-OpenAI-text-embedding-3-large embedder.
# Keep it for back-compat (the OpenAI default model still resolves to this via
# litellm), but log a warning when we hit it without a real lookup.
_FALLBACK_DIMENSIONS = 3072


def _resolve_embedding_dimensions(provider: Optional[str], model: Optional[str]) -> Optional[int]:
    """Best-effort lookup of the embedding dimensionality for a provider+model.

    Returns the dimension count if we can confidently determine it, or None
    if the model is unknown to litellm and fastembed. Defensive against
    optional-dependency and registry-schema variations — never raises.
    """
    if not provider or not model:
        return None

    provider_lower = provider.lower()
    # Strip "openai/" / "azure/" / etc. prefix from "openai/text-embedding-3-large"
    bare_model = model.split("/")[-1] if "/" in model else model
    candidates = [model, bare_model, f"{provider_lower}/{bare_model}"]

    if provider_lower == "fastembed":
        try:
            from fastembed import TextEmbedding

            for entry in TextEmbedding.list_supported_models():
                if entry.get("model") in candidates:
                    # fastembed has shipped both `dim` and `embed_dim` over time
                    dim = entry.get("dim") or entry.get("embed_dim")
                    if dim:
                        return int(dim)
        except Exception:
            pass
        # Fall through to litellm in case the model is dual-registered
        # (rare, but cheap to try).

    try:
        import litellm

        for candidate in candidates:
            info = litellm.model_cost.get(candidate)
            if info and "output_vector_size" in info:
                return int(info["output_vector_size"])
    except Exception:
        pass

    return None


class EmbeddingConfig(BaseSettings):
    """
    Manage configuration settings for embedding operations, including provider, model
    details, API configuration, and tokenizer settings.

    Public methods:
    - to_dict: Serialize the configuration settings to a dictionary.
    """

    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    # Resolved in model_post_init when not set explicitly. Was hard-defaulted
    # to 3072, which silently broke every non-OpenAI-text-embedding-3-large
    # embedder by causing a Vector(3072) / 384-dim (etc.) mismatch on first
    # write into the vector store.
    embedding_dimensions: Optional[int] = None
    embedding_endpoint: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_api_version: Optional[str] = None
    embedding_max_completion_tokens: Optional[int] = Field(
        default=8191,
        validation_alias=AliasChoices(
            "embedding_max_completion_tokens",
            "embedding_max_tokens",
            "EMBEDDING_MAX_TOKENS",
        ),
    )
    embedding_batch_size: Optional[int] = None
    huggingface_tokenizer: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def model_post_init(self, __context) -> None:
        if self.embedding_provider is None:
            from cognee.infrastructure.llm.config import get_llm_context_config

            llm_config = get_llm_context_config()
            llm_provider = llm_config.llm_provider
            if not llm_provider and llm_config.llm_model:
                llm_provider = (
                    llm_config.llm_model.split("/")[0]
                    if "/" in llm_config.llm_model
                    else llm_config.llm_model
                )

            if llm_provider == "openai":
                self.embedding_provider = "openai"
                if not self.embedding_model:
                    self.embedding_model = "openai/text-embedding-3-large"
            elif llm_provider == "ollama":
                self.embedding_provider = "ollama"
                if not self.embedding_model:
                    self.embedding_model = "nomic-embed-text"
            else:
                raise ValueError(
                    f"Embedding provider is not set. Since your LLM provider is '{llm_provider}', "
                    "we cannot silently default to OpenAI. Please set EMBEDDING_PROVIDER and EMBEDDING_MODEL explicitly."
                )

        if not self.embedding_model:
            if self.embedding_provider == "openai":
                self.embedding_model = "openai/text-embedding-3-large"
            elif self.embedding_provider == "ollama":
                self.embedding_model = "nomic-embed-text"

        if self.embedding_dimensions is None:
            derived = _resolve_embedding_dimensions(self.embedding_provider, self.embedding_model)
            if derived is not None:
                self.embedding_dimensions = derived
            else:
                raise ValueError(
                    f"Could not auto-derive embedding_dimensions for provider='{self.embedding_provider}' model='{self.embedding_model}'. "
                    "Please specify EMBEDDING_DIMENSIONS explicitly in your environment configuration."
                )

        if not self.embedding_batch_size:
            self.embedding_batch_size = 36

    def to_dict(self) -> dict:
        """
        Serialize all embedding configuration settings to a dictionary.

        Returns:
        --------

            - dict: A dictionary containing the embedding configuration settings.
        """
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_endpoint": self.embedding_endpoint,
            "embedding_api_key": self.embedding_api_key,
            "embedding_api_version": self.embedding_api_version,
            "embedding_max_completion_tokens": self.embedding_max_completion_tokens,
            "huggingface_tokenizer": self.huggingface_tokenizer,
        }


@lru_cache
def get_embedding_config():
    """
    Retrieve a cached instance of the EmbeddingConfig class.

    This function returns an instance of EmbeddingConfig with default settings. It uses
    memoization to cache the result, ensuring that subsequent calls return the same instance
    without re-initialization, improving performance and resource utilization.

    Returns:
    --------

        - EmbeddingConfig: An instance of EmbeddingConfig containing the embedding
          configuration settings.
    """
    return EmbeddingConfig()


def get_embedding_context_config() -> EmbeddingConfig:
    """Get the appropriate embedding config based on the current async context.

    Mirrors the graph/vector context-config pattern: if an ``EmbeddingConfig`` has
    been set on the ``embedding_config`` ContextVar (via
    ``set_database_global_context_variables``), return it so that different async
    tasks, threads and processes can use different embedding configurations.
    Otherwise fall back to the cached global config.
    """
    from cognee.context_global_variables import embedding_config

    return embedding_config.get() or get_embedding_config()
