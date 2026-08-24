import pytest

from cognee.infrastructure.llm.config import LLMConfig


def test_strip_quotes_from_strings():
    """
    Test if the LLMConfig.strip_quotes_from_strings model validator behaves as expected.
    """
    config = LLMConfig(
        # Strings with surrounding double quotes ("value" → value)
        llm_api_key='"double_value"',
        # Strings with surrounding single quotes ('value' → value)
        llm_endpoint="'single_value'",
        # Strings without quotes (value → value)
        llm_api_version="no_quotes_value",
        # Empty quoted strings ("" → empty string)
        fallback_model='""',
        # None values (should remain None)
        baml_llm_api_key=None,
        # Mixed quotes ("value' → unchanged)
        fallback_endpoint="\"mixed_quote'",
        # Strings with internal quotes ("internal\"quotes" → internal"quotes")
        baml_llm_model='"internal"quotes"',
    )

    # Strings with surrounding double quotes ("value" → value)
    assert config.llm_api_key == "double_value"

    # Strings with surrounding single quotes ('value' → value)
    assert config.llm_endpoint == "single_value"

    # Strings without quotes (value → value)
    assert config.llm_api_version == "no_quotes_value"

    # Empty quoted strings ("" → empty string)
    assert config.fallback_model == ""

    # None values (should remain None)
    assert config.baml_llm_api_key is None

    # Mixed quotes ("value' → unchanged)
    assert config.fallback_endpoint == "\"mixed_quote'"

    # Strings with internal quotes ("internal\"quotes" → internal"quotes")
    assert config.baml_llm_model == 'internal"quotes'


def test_strip_quotes_generalized_to_all_string_fields(monkeypatch):
    """
    Quote-stripping applies to every string field, not just a hard-coded allow-list.

    ``transcription_model`` was not part of the original 14-field allow-list, so it
    exercises the generalized behavior.
    """
    for var in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    config = LLMConfig(transcription_model='"whisper-1"', _env_file=None)
    assert config.transcription_model == "whisper-1"


def test_infer_provider_from_model(monkeypatch):
    """
    llm_provider is inferred from the llm_model prefix when it is not set explicitly.
    """
    for var in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    # Only the model is provided -> provider inferred from the litellm-style prefix.
    config = LLMConfig(llm_model="anthropic/claude-3-5-sonnet-20241022", _env_file=None)
    assert config.llm_provider == "anthropic"


def test_explicit_provider_takes_precedence(monkeypatch):
    """
    An explicitly set llm_provider is never overridden by inference.
    """
    for var in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    config = LLMConfig(
        llm_provider="custom",
        llm_model="openrouter/google/gemini-2.0-flash-lite",
        _env_file=None,
    )
    assert config.llm_provider == "custom"


def test_env_provider_takes_precedence_over_inference(monkeypatch):
    """
    A provider set via LLM_PROVIDER (env) suppresses inference, just like a kwarg.

    This is the real-world back-compat path: existing configs that pair
    LLM_PROVIDER="custom" with an unsupported-prefix model (e.g. OpenRouter)
    must keep working and must not raise ProviderNotDeducibleError. It exercises
    the env source (not a kwarg), confirming env-set fields land in
    model_fields_set so inference is skipped.
    """
    monkeypatch.setenv("LLM_PROVIDER", "custom")
    monkeypatch.setenv("LLM_MODEL", "openrouter/google/gemini-2.0-flash-lite")

    config = LLMConfig(_env_file=None)
    assert config.llm_provider == "custom"


def test_provider_unchanged_without_prefix(monkeypatch):
    """
    A model without a '/' prefix leaves the default provider untouched.
    """
    for var in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    config = LLMConfig(llm_model="gpt-4o", _env_file=None)
    assert config.llm_provider == "openai"


def test_default_config_provider_consistent(monkeypatch):
    """
    Defaults remain backward compatible (openai provider, openai/gpt-5-mini model).
    """
    for var in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    config = LLMConfig(_env_file=None)
    assert config.llm_provider == "openai"
    assert config.llm_model == "openai/gpt-5-mini"


def test_unknown_provider_prefix_raises(monkeypatch):
    """
    An unrecognized model prefix raises rather than guessing a bad provider.

    e.g. OpenRouter models use LLM_PROVIDER="custom"; if only the model is set we
    must not guess provider="openrouter" (which cognee cannot dispatch) nor
    silently fall back. Per maintainer guidance on the issue, we raise and tell
    the user to set the provider explicitly.
    """
    from cognee.infrastructure.llm.exceptions import ProviderNotDeducibleError

    for var in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ProviderNotDeducibleError):
        LLMConfig(llm_model="openrouter/google/gemini-2.0-flash-lite", _env_file=None)


def test_ollama_partial_embedding_env_does_not_raise(monkeypatch):
    """
    The Ollama validator no longer validates embedding env vars (that belongs to
    EmbeddingConfig), so partial embedding env no longer blocks LLMConfig.
    """
    for var in (
        "LLM_MODEL",
        "LLM_ENDPOINT",
        "LLM_API_KEY",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")  # only one embedding var set

    config = LLMConfig(llm_provider="ollama", _env_file=None)
    assert config.llm_provider == "ollama"


def test_ollama_partial_llm_env_still_raises(monkeypatch):
    """
    The Ollama validator still enforces that LLM env vars are all-or-nothing.
    """
    for var in ("LLM_ENDPOINT", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_MODEL", "ollama/llama3.1:8b")  # only one LLM var set

    with pytest.raises(ValueError):
        LLMConfig(llm_provider="ollama", _env_file=None)


def test_instructor_mode_table_and_adapter_wiring():
    """
    Instructor-mode defaults come from one central table, and adapters read from
    it (single source of truth) instead of hard-coding their own default.
    """
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.instructor_modes import (
        get_instructor_mode,
    )

    # Table values match the historical per-adapter defaults.
    assert get_instructor_mode("openai") == "json_schema_mode"
    assert get_instructor_mode("anthropic") == "anthropic_tools"
    # ollama DELIBERATELY diverges from its historical "json_mode" default.
    # json_mode sends response_format=json_object, so the pydantic schema reaches
    # the model as prompt text only and never as a decoder constraint. Measured
    # against llama3.1:8b with max_retries=0: json_mode returned 2/5 valid
    # KnowledgeGraph objects on the first attempt, json_schema_mode returned 5/5.
    assert get_instructor_mode("ollama") == "json_schema_mode"
    # An unknown provider fails loudly rather than silently defaulting.
    with pytest.raises(KeyError):
        get_instructor_mode("totally-unknown-provider")

    # Adapters now derive their default from the table. OpenAIAdapter is used
    # here because it has no optional third-party dependency to import.
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
        OpenAIAdapter,
    )

    assert OpenAIAdapter.default_instructor_mode == get_instructor_mode("openai")


def _clear_sampling_env(monkeypatch):
    # LLM_ENDPOINT and LLM_API_KEY belong here even though no test reads them:
    # cognee loads .env into os.environ at import, and pydantic-settings fills
    # any field not passed as a kwarg from the environment - so an ambient
    # LLM_ENDPOINT/LLM_API_KEY reaches LLMConfig.ensure_env_vars_for_ollama the
    # same as a kwarg would. Leaving either one set makes an ollama config
    # raise "some but not all of the required environment variables" no
    # matter what the test passes as kwargs. _env_file=None does not help: it
    # disables pydantic's dotenv reading, not the environment already
    # populated at import time.
    for var in (
        "LLM_TEMPERATURE",
        "LLM_SEED",
        "LLM_ARGS",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_ENDPOINT",
        "LLM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_unset_temperature_is_not_folded_into_llm_args(monkeypatch):
    """
    An unset llm_temperature must not be sent to the provider: the default
    model family (gpt-5) rejects any temperature other than its own default.
    """
    _clear_sampling_env(monkeypatch)

    config = LLMConfig(_env_file=None)
    assert config.llm_args is None


def test_explicit_temperature_kwarg_folds_into_llm_args(monkeypatch):
    _clear_sampling_env(monkeypatch)

    config = LLMConfig(llm_temperature=0.0, _env_file=None)
    assert config.llm_args == {"temperature": 0.0}


def test_env_temperature_folds_into_llm_args(monkeypatch):
    """
    LLM_TEMPERATURE set via env reaches llm_args (env-set fields land in
    model_fields_set). This is the fix for the field being dead config: it was
    documented as the determinism knob but never plumbed to any adapter.
    """
    _clear_sampling_env(monkeypatch)
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")

    config = LLMConfig(_env_file=None)
    assert config.llm_args == {"temperature": 0.0}


def test_llm_args_temperature_wins_over_dedicated_field(monkeypatch):
    _clear_sampling_env(monkeypatch)

    config = LLMConfig(
        llm_temperature=0.0,
        llm_args={"temperature": 0.7, "max_tokens": 1024},
        _env_file=None,
    )
    assert config.llm_args == {"temperature": 0.7, "max_tokens": 1024}


def test_seed_folds_into_llm_args_and_preserves_existing_keys(monkeypatch):
    _clear_sampling_env(monkeypatch)

    config = LLMConfig(llm_seed=42, llm_args={"max_tokens": 1024}, _env_file=None)
    assert config.llm_args == {"seed": 42, "max_tokens": 1024}


def _build_local(monkeypatch, **kwargs):
    """Build a config for a local inference server with sampling env cleared."""
    _clear_sampling_env(monkeypatch)
    defaults = {
        "llm_api_key": "test-key",
        "llm_endpoint": "http://localhost:11434/v1",
        "_env_file": None,
    }
    defaults.update(kwargs)
    return LLMConfig(**defaults)


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("ollama", "gemma4:e2b-it-qat"),
        ("llama_cpp", "some-model"),
        ("custom", "lm_studio/qwen2.5-7b"),
    ],
)
def test_local_provider_folds_unset_temperature(monkeypatch, provider, model):
    """
    On a local inference server an unset llm_temperature must still reach the
    provider. The gpt-5 restriction that gates the fold does not apply there,
    and without this the model's own default applies (1.0 for several Ollama
    models) even though docs/ollama_models.md documents 0.0 for extraction.
    """
    config = _build_local(monkeypatch, llm_provider=provider, llm_model=model)
    assert config.llm_args == {"temperature": 0.0}


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "openai/gpt-5-mini"),
        ("custom", "hosted_vllm/meta-llama/Llama-3-70B"),
        ("custom", "vllm/some-model"),
    ],
)
def test_non_local_provider_still_omits_unset_temperature(monkeypatch, provider, model):
    """The gpt-5 guard is untouched for hosted providers, vLLM included."""
    config = _build_local(monkeypatch, llm_provider=provider, llm_model=model)
    assert config.llm_args is None


def test_explicit_temperature_wins_on_local_provider(monkeypatch):
    config = _build_local(
        monkeypatch,
        llm_provider="ollama",
        llm_model="gemma4:e2b-it-qat",
        llm_temperature=0.7,
    )
    assert config.llm_args == {"temperature": 0.7}


def test_llm_args_temperature_wins_over_local_default(monkeypatch):
    """LLM_ARGS keeps precedence over the folded local default."""
    config = _build_local(
        monkeypatch,
        llm_provider="ollama",
        llm_model="gemma4:e2b-it-qat",
        llm_args={"temperature": 0.7, "max_tokens": 1024},
    )
    assert config.llm_args == {"temperature": 0.7, "max_tokens": 1024}


def test_known_providers_match_enum():
    """
    KNOWN_LLM_PROVIDERS must stay aligned with the LLMProvider dispatch enum so
    inference never yields a provider the client cannot construct.
    """
    from cognee.infrastructure.llm.config import KNOWN_LLM_PROVIDERS
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
        LLMProvider,
    )

    assert KNOWN_LLM_PROVIDERS == {provider.value for provider in LLMProvider}


class TestEnsureEnvVarsForOllama:
    """The ollama all-or-nothing guard must judge the resolved config, not os.environ.

    COG-6293: this validator used to check os.environ directly, so a config
    built entirely from kwargs (as every case below does) was invisible to
    it, and a caller's unrelated ambient .env could trip or silence it by
    accident. It now reads self.llm_model / self.llm_endpoint / self.llm_api_key
    after pydantic-settings has already merged env and kwargs into them.
    """

    def test_all_three_provided_as_kwargs_is_fine(self, monkeypatch):
        for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_ENDPOINT", "LLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        config = LLMConfig(
            _env_file=None,
            llm_provider="ollama",
            llm_model="gemma4:e2b-it-qat",
            llm_endpoint="http://localhost:11434/v1",
            llm_api_key="test-key",
        )

        assert config.llm_model == "gemma4:e2b-it-qat"

    def test_none_provided_is_fine(self, monkeypatch):
        """Only llm_provider set: no partial state to complain about."""
        for var in ("LLM_MODEL", "LLM_ENDPOINT", "LLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        LLMConfig(_env_file=None, llm_provider="ollama")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"llm_endpoint": "http://localhost:11434/v1"},
            {"llm_api_key": "test-key"},
            {"llm_model": "gemma4:e2b-it-qat"},
        ],
    )
    def test_exactly_one_provided_raises(self, monkeypatch, kwargs):
        for var in ("LLM_MODEL", "LLM_ENDPOINT", "LLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(ValueError, match="some but not all"):
            LLMConfig(_env_file=None, llm_provider="ollama", **kwargs)

    def test_ambient_env_unrelated_to_the_kwargs_still_raises(self, monkeypatch):
        """An LLM_API_KEY left over from the caller's own .env must still count.

        Guards against overcorrecting into ignoring env entirely: the fields
        are resolved from env-or-kwarg by pydantic-settings before this
        validator runs, so an ambient LLM_API_KEY combined with a kwarg-only
        llm_model is exactly the "some but not all" case, and must still raise.
        """
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_ENDPOINT", raising=False)
        monkeypatch.setenv("LLM_API_KEY", "from-ambient-env")

        with pytest.raises(ValueError, match="some but not all"):
            LLMConfig(_env_file=None, llm_provider="ollama", llm_model="gemma4:e2b-it-qat")

    def test_non_ollama_provider_is_never_checked(self, monkeypatch):
        for var in ("LLM_MODEL", "LLM_ENDPOINT", "LLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        LLMConfig(_env_file=None, llm_provider="openai", llm_api_key="test-key")

    def test_llm_model_explicitly_set_to_the_default_value_still_counts_as_set(self, monkeypatch):
        """A kwarg/env value that happens to match the field default is still 'set'.

        llm_model's default ("openai/gpt-5-mini") is a real, non-blank model
        id, so a plain non-blank check can't tell "explicitly configured,
        coincidentally matches the default" apart from "left unset". This
        must key off model_fields_set instead, the same way
        infer_provider_from_model already does for llm_provider.
        """
        for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_ENDPOINT", "LLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        default_model = LLMConfig.model_fields["llm_model"].default
        config = LLMConfig(
            _env_file=None,
            llm_provider="ollama",
            llm_model=default_model,
            llm_endpoint="http://localhost:11434/v1",
            llm_api_key="test-key",
        )

        assert config.llm_model == default_model

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"llm_endpoint": "   "},
            {"llm_api_key": "   "},
            {"llm_model": "   "},
        ],
    )
    def test_whitespace_only_value_counts_as_unset(self, monkeypatch, kwargs):
        """A whitespace-only value carries no real configuration, so it must not
        satisfy the "this field is set" side of the all-or-nothing check."""
        for var in ("LLM_MODEL", "LLM_ENDPOINT", "LLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        config_kwargs = {
            "llm_endpoint": "http://localhost:11434/v1",
            "llm_api_key": "test-key",
            "llm_model": "gemma4:e2b-it-qat",
        }
        config_kwargs.update(kwargs)

        with pytest.raises(ValueError, match="some but not all"):
            LLMConfig(_env_file=None, llm_provider="ollama", **config_kwargs)
