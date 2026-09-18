from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.exceptions import CogneeConfigurationError
from cognee.shared.logging_utils import get_logger

logger = get_logger("embedding_config")


# Hard fallback when neither litellm nor fastembed knows the model. This used
# to be the unconditional default and was the source of the silent
# "Vector(3072)" mismatch on every non-OpenAI-text-embedding-3-large embedder.
# Keep it for back-compat (the OpenAI default model still resolves to this via
# litellm), but log a warning when we hit it without a real lookup.
_FALLBACK_DIMENSIONS = 3072

DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-large"

# What embeddings run on when nothing is configured and no usable LLM key
# exists to reuse for the OpenAI default: a local CPU model (`fastembed`
# extra), matching the local GLiNER extractor cognify picks in that state.
# bge-small is the smallest download in the fastembed registry (67 MB) that
# is a real retrieval model; its vector size is read from the registry so
# the model is the only thing to change here.
DEFAULT_LOCAL_EMBEDDING_PROVIDER = "fastembed"
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class KeylessEmbedderNotInstalledError(CogneeConfigurationError):
    """No LLM key is configured and the local embedder's package is missing."""

    def __init__(self):
        super().__init__(
            "No LLM API key is configured, so embeddings would run on the local fastembed "
            f"model {DEFAULT_LOCAL_EMBEDDING_MODEL}, but the `fastembed` package (a cognee "
            "dependency) is not importable. Reinstall it with: pip install fastembed, or set "
            "LLM_API_KEY to embed with the OpenAI default.",
            "KeylessEmbedderNotInstalledError",
        )


def _resolve_embedding_dimensions(provider: str | None, model: str | None) -> int | None:
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
            logger.debug("Ignoring exception in _resolve_embedding_dimensions", exc_info=True)
        # Fall through to litellm in case the model is dual-registered
        # (rare, but cheap to try).

    try:
        import litellm

        for candidate in candidates:
            info = litellm.model_cost.get(candidate)
            if info and "output_vector_size" in info:
                return int(info["output_vector_size"])
    except Exception:
        logger.debug("Ignoring exception in _resolve_embedding_dimensions", exc_info=True)

    return None


class EmbeddingConfig(BaseSettings):
    """
    Manage configuration settings for embedding operations, including provider, model
    details, API configuration, and tokenizer settings.

    Public methods:
    - to_dict: Serialize the configuration settings to a dictionary.
    """

    embedding_provider: str | None = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str | None = DEFAULT_EMBEDDING_MODEL
    # Resolved in model_post_init when not set explicitly. Was hard-defaulted
    # to 3072, which silently broke every non-OpenAI-text-embedding-3-large
    # embedder by causing a Vector(3072) / 384-dim (etc.) mismatch on first
    # write into the vector store.
    embedding_dimensions: int | None = None
    # Also accepted as EMBEDDING_API_BASE — the name the litellm/OpenAI
    # ecosystem uses (issue #4871: with only EMBEDDING_ENDPOINT recognized and
    # extra="allow" swallowing unknowns, a custom base set via API_BASE was
    # silently ignored and requests 404'd against api.openai.com).
    # EMBEDDING_ENDPOINT wins when both are set.
    embedding_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_ENDPOINT", "EMBEDDING_API_BASE"),
    )
    embedding_api_key: str | None = None
    embedding_api_version: str | None = None
    embedding_max_completion_tokens: int | None = 8191
    embedding_batch_size: int | None = None
    # Total data points allowed in flight to the embedding engine during indexing.
    # Concurrent embedding requests = max(1, this // embedding_batch_size).
    embedding_max_concurrent_data_points: int = 150
    huggingface_tokenizer: str | None = None
    # Some providers (e.g. NVIDIA NIM's nv-embed family) require an
    # "input_type" field in the embedding request body (typically "query" or
    # "passage"/"document"). This is not part of the OpenAI embeddings spec,
    # so it has no effect on providers that don't recognize it. Configure via
    # the EMBEDDING_INPUT_TYPE env var.
    embedding_input_type: str | None = None
    # Rate-limiting for embedding requests. Lives here (not in LLMConfig) so the
    # knobs sit with the embedding settings they govern.
    embedding_rate_limit_enabled: bool = False
    embedding_rate_limit_requests: int = 60
    embedding_rate_limit_interval: int = 60  # in seconds (default is 60 requests per minute)
    embedding_rate_limit_tokens: int = 0  # max tokens per interval (0 = disabled)
    model_config = SettingsConfigDict(env_file=".env", extra="allow", populate_by_name=True)

    def model_post_init(self, context, /) -> None:
        if self.embedding_dimensions is None:
            derived = _resolve_embedding_dimensions(self.embedding_provider, self.embedding_model)
            if derived is not None:
                self.embedding_dimensions = derived
            else:
                logger.warning(
                    "Could not auto-derive embedding_dimensions for "
                    "provider=%r model=%r. Falling back to %d. If your embedder "
                    "produces vectors of a different size, set EMBEDDING_DIMENSIONS "
                    "explicitly — otherwise the first write into the vector store "
                    "will fail with a shape mismatch.",
                    self.embedding_provider,
                    self.embedding_model,
                    _FALLBACK_DIMENSIONS,
                )
                self.embedding_dimensions = _FALLBACK_DIMENSIONS

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
            "embedding_input_type": self.embedding_input_type,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_concurrent_data_points": self.embedding_max_concurrent_data_points,
            "embedding_rate_limit_enabled": self.embedding_rate_limit_enabled,
            "embedding_rate_limit_requests": self.embedding_rate_limit_requests,
            "embedding_rate_limit_interval": self.embedding_rate_limit_interval,
        }


def embedding_settings_configured(config) -> bool:
    """True when any embedding setting was configured: provider, model, key or
    endpoint differs from the stock OpenAI default.

    Value-based on purpose: settings arrive from env vars, kwargs and
    ``cognee.config.set_embedding_*`` alike, and only the values tell the
    cases apart. The preflight and the engine factory share this predicate.
    """
    return not (
        (config.embedding_provider or "").lower() == DEFAULT_EMBEDDING_PROVIDER
        and (config.embedding_model or "") == DEFAULT_EMBEDDING_MODEL
        and not (config.embedding_api_key or "").strip()
        and not config.embedding_endpoint
    )


def resolve_embedding_defaults(config, llm_config) -> tuple[str | None, str | None, int | None]:
    """Return the ``(provider, model, dimensions)`` the embedding engine runs with.

    The OpenAI default embedder only works because ``LLM_API_KEY`` is reused
    for it. With no embedding setting configured and no usable LLM key, that
    default cannot run, so embeddings go to the local fastembed model instead — the
    embedding half of keyless ingestion (``resolve_extractor`` is the graph
    half). Any configured embedding setting, a usable LLM key, or a disabled
    preflight (``keyless_local_defaults_apply``) keeps the config exactly as
    given.
    """
    from cognee.modules.preflight import keyless_local_defaults_apply

    if not embedding_settings_configured(config) and keyless_local_defaults_apply(llm_config):
        dimensions = _resolve_embedding_dimensions(
            DEFAULT_LOCAL_EMBEDDING_PROVIDER, DEFAULT_LOCAL_EMBEDDING_MODEL
        )
        if dimensions is None:
            # The registry lookup only fails when fastembed itself is absent.
            raise KeylessEmbedderNotInstalledError()
        return DEFAULT_LOCAL_EMBEDDING_PROVIDER, DEFAULT_LOCAL_EMBEDDING_MODEL, dimensions
    return config.embedding_provider, config.embedding_model, config.embedding_dimensions


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
