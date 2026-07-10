import json
import os
from functools import lru_cache
from typing import Any, ClassVar

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from baml_py import ClientRegistry  # ty:ignore[unresolved-import]
except ImportError:
    ClientRegistry = None

# Module-level constant (not a class attribute, to avoid pydantic field detection).
_STAGE_NAMES = {"extraction", "summarization", "query"}


# Provider identifiers cognee can dispatch to. Kept in sync with the
# ``LLMProvider`` enum in
# ``.structured_output_framework.litellm_instructor.llm.get_llm_client``; a unit
# test asserts the two stay aligned. Defined here rather than imported to avoid a
# circular import, since that module imports from this one.
KNOWN_LLM_PROVIDERS = frozenset(
    {
        "openai",
        "ollama",
        "anthropic",
        "custom",
        "gemini",
        "mistral",
        "azure",
        "bedrock",
        "llama_cpp",
    }
)


class LLMConfig(BaseSettings):
    """
    Configuration settings for the LLM (Large Language Model) provider and related options.

    Public instance variables include:
    - llm_provider
    - llm_model
    - llm_endpoint
    - llm_api_key
    - llm_api_version
    - llm_temperature
    - llm_streaming
    - llm_max_completion_tokens
    - transcription_model
    - graph_prompt_path
    - llm_rate_limit_enabled
    - llm_rate_limit_requests
    - llm_rate_limit_interval

    Public methods include:
    - ensure_env_vars_for_ollama
    - to_dict
    - stage_config
    """

    structured_output_framework: str = "instructor"
    llm_instructor_mode: str = ""
    llm_provider: str = "openai"
    llm_model: str = "openai/gpt-5-mini"
    llm_endpoint: str = ""
    llm_api_key: str | None = None
    llm_api_version: str | None = None

    # Per-stage model routing (optional). Empty means fall back to the base llm_* values.
    llm_extraction_model: str = ""
    llm_extraction_provider: str = ""
    llm_extraction_endpoint: str = ""
    llm_extraction_api_key: str | None = None
    llm_extraction_api_version: str | None = None

    llm_summarization_model: str = ""
    llm_summarization_provider: str = ""
    llm_summarization_endpoint: str = ""
    llm_summarization_api_key: str | None = None
    llm_summarization_api_version: str | None = None

    llm_query_model: str = ""
    llm_query_provider: str = ""
    llm_query_endpoint: str = ""
    llm_query_api_key: str | None = None
    llm_query_api_version: str | None = None

    llm_temperature: float = 0.0
    llm_streaming: bool = False
    llm_max_completion_tokens: int = 16384

    baml_llm_provider: str = "openai"
    baml_llm_model: str = "gpt-5-mini"
    baml_llm_endpoint: str = ""
    baml_llm_api_key: str | None = None
    baml_llm_temperature: float = 0.0
    baml_llm_api_version: str = ""

    transcription_model: str = "whisper-1"
    graph_prompt_path: str = "generate_graph_prompt.txt"
    temporal_graph_prompt_path: str = "generate_event_graph_prompt.txt"
    event_entity_prompt_path: str = "generate_event_entity_prompt.txt"
    llm_rate_limit_enabled: bool = False
    llm_rate_limit_requests: int = 60
    llm_rate_limit_interval: int = 60  # in seconds (default is 60 requests per minute)
    llm_rate_limit_tokens: int = 0  # max tokens per interval (0 = disabled)

    llama_cpp_model_path: str | None = None
    llama_cpp_n_ctx: int = 2048
    llama_cpp_n_gpu_layers: int = 0
    llama_cpp_chat_format: str = "chatml"

    fallback_api_key: str = ""
    fallback_endpoint: str = ""
    fallback_model: str = ""

    llm_azure_use_managed_identity: bool = False

    llm_args: dict[str, Any] | None = None

    baml_registry: Any | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @model_validator(mode="after")
    def strip_quotes_from_strings(self) -> "LLMConfig":
        """
        Strip surrounding quotes from string fields that often arrive from
        environment variables with extra quotes (e.g., via Docker's --env-file).

        Applies to every declared string field rather than a hand-maintained
        allow-list, so newly added string fields are covered automatically.
        Only a matching pair of surrounding quotes (both ``'`` or both ``"``) is
        removed; mismatched or internal quotes are left untouched.
        """
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name, None)
            if isinstance(value, str) and len(value) >= 2:
                if value[0] == value[-1] and value[0] in ("'", '"'):
                    setattr(self, field_name, value[1:-1])

        return self

    @model_validator(mode="after")
    def infer_provider_from_model(self) -> "LLMConfig":
        """
        Infer ``llm_provider`` from the ``llm_model`` prefix when the provider was
        not set explicitly.

        LiteLLM-style model identifiers embed the provider as a prefix
        (e.g. ``"anthropic/claude-3-5-sonnet"``). When only ``LLM_MODEL`` is
        supplied, the provider is derived from that prefix so ``LLM_PROVIDER``
        becomes optional.

        Inference is intentionally conservative: it only applies when the prefix
        is a provider cognee actually supports (see ``KNOWN_LLM_PROVIDERS``).
        A prefixed model whose prefix is *not* a supported provider (e.g.
        ``"openrouter/..."``, which is configured via ``LLM_PROVIDER="custom"``)
        raises ``ProviderNotDeducibleError`` rather than silently falling back to
        a default that would fail downstream, so the user is told to set
        ``LLM_PROVIDER`` explicitly. An explicitly provided ``llm_provider``
        (keyword argument or environment variable) always takes precedence, as
        does a model without a ``"/"`` prefix.

        Runs after ``strip_quotes_from_strings`` so the model prefix is compared
        without surrounding quotes.
        """
        if "llm_provider" in self.model_fields_set:
            return self

        model = self.llm_model
        if isinstance(model, str) and "/" in model:
            prefix = model.split("/", 1)[0].strip().lower()
            if prefix in KNOWN_LLM_PROVIDERS:
                self.llm_provider = prefix
            else:
                from cognee.infrastructure.llm.exceptions import ProviderNotDeducibleError

                raise ProviderNotDeducibleError(model)

        return self

    def model_post_init(self, __context) -> None:
        """Initialize the BAML registry after the model is created."""
        # Check if BAML is selected as structured output framework but not available
        if self.structured_output_framework.lower() == "baml" and ClientRegistry is None:
            raise ImportError(
                "BAML is selected as structured output framework but not available. "
                "Please install with 'pip install cognee\"[baml]\"' to use BAML extraction features."
            )
        elif self.structured_output_framework.lower() == "baml" and ClientRegistry is not None:
            self.baml_registry = ClientRegistry()

            raw_options = {
                "model": self.baml_llm_model,
                "temperature": self.baml_llm_temperature,
                "api_key": self.baml_llm_api_key,
                "base_url": self.baml_llm_endpoint,
                "api_version": self.baml_llm_api_version,
            }

            # Note: keep the item only when the value is not None or an empty string (they would override baml default values)
            options = {k: v for k, v in raw_options.items() if v not in (None, "")}
            self.baml_registry.add_llm_client(
                name=self.baml_llm_provider, provider=self.baml_llm_provider, options=options
            )
            # Sets the primary client
            self.baml_registry.set_primary(self.baml_llm_provider)

    @model_validator(mode="after")
    def ensure_env_vars_for_ollama(self) -> "LLMConfig":
        """
        Validate required environment variables for the 'ollama' LLM provider.

        Raises ValueError if some required environment variables are set without the others.
        Only checks are performed when 'llm_provider' is set to 'ollama'.

        Returns:
        --------

            - 'LLMConfig': The instance of LLMConfig after validation.
        """

        if self.llm_provider != "ollama":
            # Skip checks unless provider is "ollama"
            return self

        def is_env_set(var_name: str) -> bool:
            """
            Check if a given environment variable is set and non-empty.

            Parameters:
            -----------

                - var_name (str): The name of the environment variable to check.

            Returns:
            --------

                - bool: True if the environment variable exists and is not empty, otherwise False.
            """
            val = os.environ.get(var_name)
            return val is not None and val.strip() != ""

        # Check LLM environment variables. Embedding env vars are intentionally
        # not validated here: that is EmbeddingConfig's responsibility, and
        # reaching into embedding settings from LLMConfig via raw os.environ was
        # an improper cross-config dependency.
        llm_env_vars = {
            "LLM_MODEL": is_env_set("LLM_MODEL"),
            "LLM_ENDPOINT": is_env_set("LLM_ENDPOINT"),
            "LLM_API_KEY": is_env_set("LLM_API_KEY"),
        }
        if any(llm_env_vars.values()) and not all(llm_env_vars.values()):
            missing_llm = [key for key, is_set in llm_env_vars.items() if not is_set]
            raise ValueError(
                "You have set some but not all of the required environment variables "
                f"for LLM usage (LLM_MODEL, LLM_ENDPOINT, LLM_API_KEY). Missing: {missing_llm}"
            )

        return self

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the LLMConfig instance into a dictionary representation.

        Returns:
        --------

            - dict: A dictionary containing the configuration settings of the LLMConfig
              instance.
        """
        return {
            "llm_instructor_mode": self.llm_instructor_mode.lower(),
            "provider": self.llm_provider,
            "model": self.llm_model,
            "endpoint": self.llm_endpoint,
            "api_key": self.llm_api_key,
            "api_version": self.llm_api_version,
            "temperature": self.llm_temperature,
            "streaming": self.llm_streaming,
            "max_completion_tokens": self.llm_max_completion_tokens,
            "transcription_model": self.transcription_model,
            "graph_prompt_path": self.graph_prompt_path,
            "rate_limit_enabled": self.llm_rate_limit_enabled,
            "rate_limit_requests": self.llm_rate_limit_requests,
            "rate_limit_interval": self.llm_rate_limit_interval,
            "fallback_api_key": self.fallback_api_key,
            "fallback_endpoint": self.fallback_endpoint,
            "fallback_model": self.fallback_model,
            "llama_cpp_model_path": self.llama_cpp_model_path,
            "llama_cpp_n_ctx": self.llama_cpp_n_ctx,
            "llama_cpp_n_gpu_layers": self.llama_cpp_n_gpu_layers,
            "llama_cpp_chat_format": self.llama_cpp_chat_format,
            "llm_args": self.llm_args,
        }

    def stage_config(self, stage: str) -> "LLMConfig":
        """Return a copy of this config with the base llm_* fields overridden by
        any set llm_<stage>_* fields. Unset stage fields fall back to the base
        values, so a config with no stage overrides returns an equivalent config
        (single-model behavior preserved).
        """
        if stage not in _STAGE_NAMES:
            return self
        update: dict[str, Any] = {}
        for field in ("model", "provider", "endpoint", "api_key", "api_version"):
            value = getattr(self, f"llm_{stage}_{field}", None)
            if value:  # treats "" and None as unset
                update[f"llm_{field}"] = value
        if not update:
            return self
        return self.model_copy(update=update)


@lru_cache
def get_llm_config() -> LLMConfig:
    """
    Retrieve and cache the LLM configuration.

    This function returns an instance of the LLMConfig class. It leverages
    caching to ensure that repeated calls do not create new instances,
    but instead return the already created configuration object.

    Returns:
    --------

        - LLMConfig: An instance of the LLMConfig class containing the configuration for the
          LLM.
    """
    return LLMConfig()


def get_llm_context_config() -> LLMConfig:
    """Get the appropriate LLM config based on the current async context.

    Mirrors the graph/vector context-config pattern: if an ``LLMConfig`` has been
    set on the ``llm_config`` ContextVar (via
    ``set_database_global_context_variables``), return it so that different async
    tasks, threads and processes can use different LLM configurations. Otherwise
    fall back to the cached global config.
    """
    from cognee.context_global_variables import llm_config

    return llm_config.get() or get_llm_config()
