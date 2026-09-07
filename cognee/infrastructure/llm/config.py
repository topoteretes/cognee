import json
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


# Providers cognee can dispatch to. Duplicated from the ``LLMProvider`` enum
# (importing it here would be circular); a unit test keeps the two in sync.
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
        "mcp-sampling",
    }
)

# Local inference servers process requests (near-)serially, unlike cloud
# providers. One definition of the distinction, for everything that needs it:
# by first-class provider name, or by litellm model routing prefix (LM Studio
# has no first-class provider in cognee). vLLM is deliberately NOT here: it
# serves with continuous batching and absorbs concurrency like a cloud
# endpoint, so it keeps the regular settings.
LOCAL_LLM_PROVIDERS = frozenset({"ollama", "llama_cpp"})
LOCAL_LLM_MODEL_PREFIXES = ("lm_studio/",)

# Default RPM budget for local inference servers when LLM_RATE_LIMIT_REQUESTS
# is not explicitly configured; cloud providers keep the regular default of 60.
LOCAL_DEFAULT_RATE_LIMIT_REQUESTS = 10


def is_local_llm(provider: str | None, model: str | None) -> bool:
    """True when the provider/model points at a serial local inference server
    (Ollama, llama.cpp by provider; LM Studio by model prefix). vLLM counts
    as regular: continuous batching absorbs concurrency like a cloud endpoint.
    """
    if (provider or "").lower() in LOCAL_LLM_PROVIDERS:
        return True
    return (model or "").lower().startswith(LOCAL_LLM_MODEL_PREFIXES)


def _apply_local_rate_limit_default(config: "LLMConfig") -> "LLMConfig":
    """Apply the local-server RPM default to ``config`` unless it was set explicitly.

    Lives outside the class so both the ``default_local_rate_limit_budget``
    validator and ``stage_config`` can use it: a ``@model_validator`` is a
    descriptor proxy on the class, not a plain callable.
    """
    if "llm_rate_limit_requests" in config.model_fields_set:
        return config

    if is_local_llm(config.llm_provider, config.llm_model):
        config.llm_rate_limit_requests = LOCAL_DEFAULT_RATE_LIMIT_REQUESTS

    return config


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
    - llm_answer_streaming
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

    # litellm_native (default): plain litellm two-path structured output —
    # schema-native response_format when the model supports it, prompted-JSON
    # fallback otherwise. No instructor in the call path, which is a
    # prerequisite for removing the instructor dependency. Exact token capture
    # works on this path via the adapter's explicit _raw_response attachment.
    # Set STRUCTURED_OUTPUT_FRAMEWORK=instructor (or baml) to opt back.
    structured_output_framework: str = "litellm_native"
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
    llm_seed: int | None = None
    llm_streaming: bool = False

    # Stream answer tokens out of the plain-text completion path so a caller can
    # render them as they arrive (env LLM_ANSWER_STREAMING). Off by default: the
    # returned value is identical either way, so enabling it changes nothing for
    # a caller that is not consuming a token sink.
    #
    # Deliberately NOT llm_streaming above, which is a different, older flag:
    # that one is unread by OpenAI/Azure, absent from every other provider, and
    # on Bedrock injects stream=True into the instructor path where nothing
    # consumes a stream. It is also part of the adapter LRU cache key, so
    # flipping it changes adapter identity.
    llm_answer_streaming: bool = False

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
    image_transcription_prompt_path: str = "transcribe_image_prompt.txt"
    image_transcription_max_completion_tokens: int = 1024
    image_transcription_reasoning_effort: str = "low"
    llm_rate_limit_enabled: bool = False
    # Default 60 requests per interval; local inference servers get
    # LOCAL_DEFAULT_RATE_LIMIT_REQUESTS instead (see default_local_rate_limit_budget).
    llm_rate_limit_requests: int = 60
    llm_rate_limit_interval: int = 60  # in seconds (default is 60 requests per minute)
    llm_rate_limit_tokens: int = 0  # max tokens per interval (0 = disabled)
    # When the provider reports a rate limit, warn and switch on the RPM limiter
    # (with the llm_rate_limit_requests/interval budget) for the rest of the process.
    auto_rate_limit: bool = True

    llama_cpp_model_path: str | None = None
    llama_cpp_n_ctx: int = 2048
    llama_cpp_n_gpu_layers: int = 0
    llama_cpp_chat_format: str = "chatml"
    ollama_num_ctx: int = 2048

    fallback_api_key: str = ""
    fallback_endpoint: str = ""
    fallback_model: str = ""

    llm_azure_use_managed_identity: bool = False

    llm_args: dict[str, Any] | None = None

    baml_registry: Any | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @model_validator(mode="before")
    @classmethod
    def blank_llm_args_is_unset(cls, values: Any) -> Any:
        """
        Treat a blank ``LLM_ARGS`` as unset rather than a validation error.

        A ``.env`` written from an empty variable — ``LLM_ARGS=`` — reaches this
        dict field as ``""`` and raises ``ValidationError`` while
        ``cognee/__init__`` is still importing, so the process dies before it can
        report anything useful. An empty value cannot express any arguments, so
        read it as "none given".
        """
        if isinstance(values, dict):
            for key in ("llm_args", "LLM_ARGS"):
                value = values.get(key)
                if isinstance(value, str) and not value.strip():
                    values[key] = None

        return values

    @model_validator(mode="after")
    def strip_quotes_from_strings(self) -> "LLMConfig":
        """
        Strip a matching pair of surrounding quotes from every string field.

        Such quotes commonly arrive from env vars passed via Docker's
        ``--env-file``. Covering all declared string fields (not an allow-list)
        means new fields are handled automatically; only a matching outer pair
        of quotes is removed, so internal quotes are left untouched.
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
        Infer ``llm_provider`` from the ``llm_model`` prefix when it was not set
        explicitly, so ``LLM_PROVIDER`` is optional for litellm-style model ids
        (e.g. ``"anthropic/claude-3-5-sonnet"`` -> ``anthropic``).

        An explicit ``llm_provider`` (kwarg or env var) always wins, as does a
        model with no ``"/"`` prefix. A prefix that is not a supported provider
        (e.g. ``"openrouter/..."``, which needs ``LLM_PROVIDER="custom"``) raises
        ``ProviderNotDeducibleError`` instead of silently defaulting to something
        that would fail downstream. Runs after ``strip_quotes_from_strings`` so
        the prefix is already unquoted.
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

    @model_validator(mode="after")
    def fold_sampling_params_into_llm_args(self) -> "LLMConfig":
        """
        Fold ``llm_temperature`` / ``llm_seed`` into ``llm_args``, the dict every
        adapter merges into each completion call — without this fold the two
        fields are read by nothing and never reach the provider.

        ``llm_temperature`` is folded when set explicitly (env var or kwarg),
        or when the model runs on a local inference server. The gate exists
        because the default model family (gpt-5) rejects any temperature other
        than the provider default, so an unset field must not silently send
        ``0.0`` there. That restriction is specific to the hosted OpenAI
        reasoning models: local servers (Ollama, llama.cpp, LM Studio) accept
        the field, and leaving it unfolded means extraction silently runs at
        whatever the model itself defaults to (1.0 for several Ollama models)
        instead of the deterministic ``0.0`` that ``docs/ollama_models.md``
        documents. Runs after ``infer_provider_from_model`` so the provider is
        already resolved. Keys given directly in ``LLM_ARGS`` win over the
        dedicated fields.
        """
        folded: dict[str, Any] = {}
        if "llm_temperature" in self.model_fields_set or is_local_llm(
            self.llm_provider, self.llm_model
        ):
            folded["temperature"] = self.llm_temperature
        if self.llm_seed is not None:
            folded["seed"] = self.llm_seed
        if folded:
            self.llm_args = {**folded, **(self.llm_args or {})}

        return self

    @model_validator(mode="after")
    def default_local_rate_limit_budget(self) -> "LLMConfig":
        """
        Give local inference servers a smaller default RPM budget.

        Serial local servers (Ollama, LM Studio, llama.cpp) process requests
        (near-)serially, so when the rate limiter engages, the regular cloud
        default of 60 requests per interval would still flood them. An
        explicitly configured ``LLM_RATE_LIMIT_REQUESTS`` always wins. Runs
        after ``infer_provider_from_model`` so the provider is already
        resolved.
        """
        return _apply_local_rate_limit_default(self)

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

        # Judge the resolved config, not os.environ. pydantic-settings has
        # already merged env vars and constructor kwargs into these fields by
        # the time an "after" validator runs, so checking os.environ here
        # would ignore a config built entirely from kwargs (as every test in
        # test_llm_config.py does) and would also stay fooled by a caller's
        # ambient .env that has nothing to do with the config under
        # construction. See COG-6293.
        #
        # llm_endpoint/llm_api_key default to blank ("" / None), so "has a
        # non-blank value" alone tells us whether they were configured.
        # llm_model defaults to a real model id ("openai/gpt-5-mini"), so the
        # same non-blank check can't tell "configured, happens to match the
        # default" from "left unset" - that also needs `model_fields_set`
        # (populated by pydantic-settings whether the value came from a kwarg
        # or an env var, even when it matches the default).
        def _is_configured(value: str | None) -> bool:
            return isinstance(value, str) and value.strip() != ""

        llm_env_vars = {
            "LLM_MODEL": "llm_model" in self.model_fields_set and _is_configured(self.llm_model),
            "LLM_ENDPOINT": _is_configured(self.llm_endpoint),
            "LLM_API_KEY": _is_configured(self.llm_api_key),
        }
        if any(llm_env_vars.values()) and not all(llm_env_vars.values()):
            missing_llm = [key for key, is_set in llm_env_vars.items() if not is_set]
            raise ValueError(
                "You have set some but not all of the required environment variables "
                f"for LLM usage (LLM_MODEL, LLM_ENDPOINT, LLM_API_KEY). Missing: {missing_llm}"
            )

        # Check model support matrix if a model is configured
        if self.llm_model:
            from cognee.infrastructure.llm.ollama_support import check_model_support

            check_model_support(self.llm_model)

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
            "seed": self.llm_seed,
            "streaming": self.llm_streaming,
            "answer_streaming": self.llm_answer_streaming,
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
            "ollama_num_ctx": self.ollama_num_ctx,
            "llm_args": self.llm_args,
        }

    def stage_config(self, stage: str) -> "LLMConfig":
        """Return a copy of this config with the base llm_* fields overridden by
        any set llm_<stage>_* fields. Unset stage fields fall back to the base
        values, so a config with no stage overrides returns an equivalent config
        (single-model behavior preserved).

        ``model_copy`` does not re-run validators, so the provider-dependent
        defaults on the copy would still be the ones derived for the *base*
        provider. Routing a stage to a different provider is the whole point of
        this method, so those defaults are recomputed below.
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

        stage_config = self.model_copy(update=update)

        # Re-derive the RPM default from the stage's own provider.
        # infer_provider_from_model is not re-run: llm_provider is always in
        # model_fields_set by this point (set explicitly, or assigned by that
        # validator on the base config), so it would return early every time.
        # ensure_env_vars_for_ollama is not re-run either, because it validates
        # the environment rather than deriving a default, and running it here
        # would move where a misconfiguration is raised.
        return _apply_local_rate_limit_default(stage_config)


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
