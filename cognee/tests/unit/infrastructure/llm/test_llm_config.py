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
    # Unknown providers fall back to the default.
    assert get_instructor_mode("totally-unknown-provider") == "json_mode"

    # Adapters now derive their default from the table. OpenAIAdapter is used
    # here because it has no optional third-party dependency to import.
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
        OpenAIAdapter,
    )

    assert OpenAIAdapter.default_instructor_mode == get_instructor_mode("openai")


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
