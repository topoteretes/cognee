import os
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.exceptions import CogneeConfigurationError
from cognee.shared.logging_utils import get_logger
from cognee.shared.overload_policy import OverloadPolicy, PacingBudget

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
# pplx-embed-v1-0.6b (0.7 GB, 1024 dimensions, multilingual) replaced bge-small
# (67 MB, 384 dimensions, English) and should provide better accuracy. It is not in
# fastembed's own list: embeddings/fastembed_models.py registers it. Its vector
# size is read from the registry. Datasets built with the previous default are
# re-embedded by the reembed_local_default_embedding_model data migration.
DEFAULT_LOCAL_EMBEDDING_PROVIDER = "fastembed"
DEFAULT_LOCAL_EMBEDDING_MODEL = "perplexity-ai/pplx-embed-v1-0.6b"
# The chunk-token cap of the local default model when EMBEDDING_MAX_COMPLETION_TOKENS
# is not set. The model reads 32,000 tokens, so without this chunks would grow to
# DEFAULT_EMBEDDING_INPUT_CAP (4096), which a CPU embeds in about 14 s and 4 GB of
# memory each (512 tokens: about 1 s). 512 is the budget the previous default had.
DEFAULT_LOCAL_EMBEDDING_INPUT_CAP = 512

# Local embedding servers serve requests (near-)serially, so their RPM default is
# lower when EMBEDDING_RATE_LIMIT_REQUESTS is not set: by provider, or by litellm
# model routing prefix. fastembed runs in-process, not on a server, so it is not
# one of them.
LOCAL_EMBEDDING_PROVIDERS = frozenset({"ollama"})
LOCAL_EMBEDDING_MODEL_PREFIXES = ("ollama/", "lm_studio/")
LOCAL_DEFAULT_EMBEDDING_RATE_LIMIT_REQUESTS = 1000
# A local server works through its queue one or a few requests at a time: Ollama
# reached full throughput at any cap from 1 to 64 in the same benchmark. A short
# queue keeps every request far from the 60 s timeout, even if the server slows down.
LOCAL_DEFAULT_EMBEDDING_MAX_CONCURRENT_REQUESTS = 4
# With an explicit endpoint, typically a self-hosted server such as TEI, whose default
# queue holds 512 texts: 16 requests of 32. A provider's public API serves more at once.
DEFAULT_EMBEDDING_MAX_CONCURRENT_REQUESTS = 16
HOSTED_DEFAULT_EMBEDDING_MAX_CONCURRENT_REQUESTS = 32

# How long embedding throttling lasts after the last overload error. An embedding
# server catches up within seconds (Ollama drained a queue of 128 in about 8 s in the
# embedding concurrency benchmark), and the Ollama engine's request timeout is 60 s, so requests
# already queued when throttling starts have finished or failed by the time it ends.
# The LLM side keeps its own, longer cooldown.
EMBEDDING_THROTTLE_COOLDOWN_SECONDS = 60.0


def is_local_embedding_server(provider: str | None, model: str | None) -> bool:
    """True when the embedding provider or model points at a local inference server."""
    if (provider or "").lower() in LOCAL_EMBEDDING_PROVIDERS:
        return True
    return (model or "").lower().startswith(LOCAL_EMBEDDING_MODEL_PREFIXES)


def default_max_concurrent_requests(
    provider: str | None, model: str | None, endpoint: str | None
) -> int:
    """The embedding request cap for where requests go, when none is set.

    Without an endpoint, requests go to the provider's public API; with one, to a
    server someone chose, most likely their own.
    """
    if is_local_embedding_server(provider, model):
        return LOCAL_DEFAULT_EMBEDDING_MAX_CONCURRENT_REQUESTS
    if not endpoint:
        return HOSTED_DEFAULT_EMBEDDING_MAX_CONCURRENT_REQUESTS
    return DEFAULT_EMBEDDING_MAX_CONCURRENT_REQUESTS


# Embedding settings that no longer exist, each with the setting that replaced it.
REMOVED_EMBEDDING_SETTINGS = {
    "EMBEDDING_MAX_CONCURRENT_DATA_POINTS": "EMBEDDING_MAX_CONCURRENT_REQUESTS",
}


def refuse_removed_embedding_settings(values) -> None:
    """Raise when a removed embedding setting is supplied, naming its replacement.

    Checks the given keys (constructor input, .env extras, the config API) and the
    process environment, where pydantic-settings never routes undeclared names.
    """
    given = {str(key).upper() for key in values} if isinstance(values, dict) else set()
    for removed, replacement in REMOVED_EMBEDDING_SETTINGS.items():
        if removed in given or removed in os.environ:
            raise CogneeConfigurationError(
                f"{removed} was removed; use {replacement} instead.",
                remediation=(
                    f"Remove {removed}. {replacement} counts embedding requests in "
                    "flight for the whole process, not data points, so do not copy the "
                    "old value; the default (64) suits most deployments."
                ),
            )


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


@lru_cache
def _resolve_embedding_dimensions(provider: str | None, model: str | None) -> int | None:
    """Best-effort lookup of the embedding dimensionality for a provider+model.

    Returns the dimension count if we can confidently determine it, or None
    if the model is unknown to litellm and fastembed. Defensive against
    optional-dependency and registry-schema variations — never raises.

    Cached per (provider, model): the fastembed and litellm registries are static
    for the life of the process, and every embedding-engine lookup resolves this.
    """
    if not provider or not model:
        return None

    provider_lower = provider.lower()
    # Strip "openai/" / "azure/" / etc. prefix from "openai/text-embedding-3-large"
    bare_model = model.split("/")[-1] if "/" in model else model
    candidates = [model, bare_model, f"{provider_lower}/{bare_model}"]

    if provider_lower == "fastembed":
        try:
            from cognee.infrastructure.databases.vector.embeddings.fastembed_models import (
                fastembed_text_models,
            )

            for entry in fastembed_text_models():
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
    # Cap on the tokens a chunk (or anything else) is embedded with. Unset means
    # DEFAULT_EMBEDDING_INPUT_CAP (4096); the engine lowers the effective limit to
    # the model's own input limit when that is known and smaller (see
    # embeddings/input_limit.py).
    embedding_max_completion_tokens: int | None = Field(default=None, gt=0)
    embedding_batch_size: int | None = None
    # Embedding provider requests in flight at once, per process and event loop,
    # shared by every caller; also the number of indexing write batches admitted
    # at once. At least 1: it bounds memory too, so there is no "unlimited".
    # Unless set, default_max_concurrent_requests picks it by where requests go.
    embedding_max_concurrent_requests: int = Field(
        default=DEFAULT_EMBEDDING_MAX_CONCURRENT_REQUESTS, ge=1
    )
    huggingface_tokenizer: str | None = None
    # Some providers (e.g. NVIDIA NIM's nv-embed family) require an
    # "input_type" field in the embedding request body (typically "query" or
    # "passage"/"document"). This is not part of the OpenAI embeddings spec,
    # so it has no effect on providers that don't recognize it. Configure via
    # the EMBEDDING_INPUT_TYPE env var.
    embedding_input_type: str | None = None
    # Wire format of embedding responses. Unset: base64 on OpenAI's own API (much
    # cheaper to parse), float everywhere else. "base64" opts any provider in except
    # fastembed and ollama, which have no wire format; the first reply shows whether the
    # route honours it. "float" opts out everywhere. See embeddings/encoding.py.
    embedding_encoding_format: Literal["base64", "float"] | None = None
    # Rate-limiting for embedding requests. Lives here (not in LLMConfig) so the
    # knobs sit with the embedding settings they govern.
    embedding_rate_limit_enabled: bool = False
    # Throttling rate: the lowest paid tiers of OpenAI, Azure OpenAI and Gemini allow
    # 1M embedding tokens a minute, about 100 requests of 32 chunks of ~300 tokens.
    # Throttling lapses 60 s after the last overload, so a low rate costs little.
    # A local server gets LOCAL_DEFAULT_EMBEDDING_RATE_LIMIT_REQUESTS.
    embedding_rate_limit_requests: int = 100
    embedding_rate_limit_interval: int = 60  # in seconds
    embedding_rate_limit_tokens: int = 0  # max tokens per interval (0 = disabled)
    model_config = SettingsConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def refuse_removed_settings(cls, values):
        # extra="allow" would otherwise keep a removed setting silently.
        refuse_removed_embedding_settings(values)
        return values

    def model_post_init(self, context, /) -> None:
        # The embedding worker reports its own vector size; left unset, the engine
        # takes it from the worker instead of from a registry or the fallback.
        worker_reports_size = (self.embedding_provider or "").lower() == "embedding_worker"
        if self.embedding_dimensions is None and not worker_reports_size:
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

        # 32 is the most inputs one request may carry on common self-hosted
        # servers (Text Embeddings Inference answers 422 above it), and the
        # difference from the old 36 is not measurable on hosted providers.
        if not self.embedding_batch_size:
            self.embedding_batch_size = 32

        # The rate and cap defaults depend on where requests go; an explicit value always wins.
        if "embedding_rate_limit_requests" not in self.model_fields_set and (
            is_local_embedding_server(self.embedding_provider, self.embedding_model)
        ):
            self.embedding_rate_limit_requests = LOCAL_DEFAULT_EMBEDDING_RATE_LIMIT_REQUESTS
        if "embedding_max_concurrent_requests" not in self.model_fields_set:
            self.embedding_max_concurrent_requests = default_max_concurrent_requests(
                self.embedding_provider, self.embedding_model, self.embedding_endpoint
            )

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
            "embedding_encoding_format": self.embedding_encoding_format,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_concurrent_requests": self.embedding_max_concurrent_requests,
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


def keyless_embedding_defaults_apply(config, llm_config) -> bool:
    """True when embeddings run on the local fastembed default instead of the config.

    The OpenAI default embedder only works because ``LLM_API_KEY`` is reused
    for it. With no embedding setting configured and no usable LLM key, that
    default cannot run, so embeddings go to the local fastembed model instead — the
    embedding half of keyless ingestion (``resolve_extractor`` is the graph
    half). Any configured embedding setting, a usable LLM key, or a disabled
    preflight (``keyless_local_defaults_apply``) keeps the config exactly as
    given.
    """
    from cognee.modules.preflight import keyless_local_defaults_apply

    return not embedding_settings_configured(config) and keyless_local_defaults_apply(llm_config)


def resolve_embedding_names(config, llm_config) -> tuple[str | None, str | None]:
    """The ``(provider, model)`` embeddings run with, without touching any registry.

    Pure: the telemetry settings payload reports the embedder through this, so
    it cannot raise ``KeylessEmbedderNotInstalledError`` — that stays with the
    engine, in ``resolve_embedding_defaults``.
    """
    if keyless_embedding_defaults_apply(config, llm_config):
        return DEFAULT_LOCAL_EMBEDDING_PROVIDER, DEFAULT_LOCAL_EMBEDDING_MODEL
    return config.embedding_provider, config.embedding_model


def resolve_embedding_defaults(config, llm_config) -> tuple[str | None, str | None, int | None]:
    """Return the ``(provider, model, dimensions)`` the embedding engine runs with.

    See ``keyless_embedding_defaults_apply`` for when the local default applies;
    its vector size is read from fastembed's registry, so a missing ``fastembed``
    surfaces here as ``KeylessEmbedderNotInstalledError``.
    """
    provider, model = resolve_embedding_names(config, llm_config)
    if keyless_embedding_defaults_apply(config, llm_config):
        dimensions = _resolve_embedding_dimensions(provider, model)
        if dimensions is None:
            # The registry lookup only fails when fastembed itself is absent.
            raise KeylessEmbedderNotInstalledError()
        return provider, model, dimensions
    return provider, model, config.embedding_dimensions


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


def embedding_pacing_budget() -> PacingBudget:
    """The embedding RPM budget an overload warning names."""
    config = get_embedding_config()
    return PacingBudget(
        config.embedding_rate_limit_requests,
        config.embedding_rate_limit_interval,
        "EMBEDDING_RATE_LIMIT_REQUESTS",
        "EMBEDDING_RATE_LIMIT_INTERVAL",
    )


# Process-wide overload policy consulted by embedding dispatch; separate from the
# LLM one, so overload on one side never paces the other.
embedding_overload_policy = OverloadPolicy(
    embedding_pacing_budget, cooldown_seconds=EMBEDDING_THROTTLE_COOLDOWN_SECONDS
)


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
