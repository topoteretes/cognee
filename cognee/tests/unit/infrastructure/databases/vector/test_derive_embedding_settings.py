"""Tests for LLM-provider-derived embedding settings.

cognee used to default embeddings to OpenAI regardless of LLM_PROVIDER,
silently reusing LLM_API_KEY against api.openai.com — a guaranteed
mid-cognify failure for every non-OpenAI setup. These tests pin down the
derivation matrix, the "explicit config wins" rule and the fail-fast
mismatch errors.
"""

import importlib
import types

import pytest
from cognee.infrastructure.databases.exceptions import EmbeddingProviderMismatchError
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.databases.vector.embeddings.derive_embedding_settings import (
    derive_embedding_settings,
)

# The embeddings package re-exports a function named get_embedding_engine,
# which shadows the module attribute — resolve the module explicitly.
engine_module = importlib.import_module(
    "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine"
)


# --- the pure derivation matrix ----------------------------------------------


def test_openai_keeps_current_default_behavior():
    derived = derive_embedding_settings("openai", None, "sk-test")
    assert derived["provider"] == "openai"
    assert derived["model"] == "openai/text-embedding-3-large"
    assert derived["endpoint"] is None
    assert derived["api_key"] == "sk-test"


def test_gemini_derives_gemini_embeddings_and_reuses_key():
    derived = derive_embedding_settings("gemini", None, "gm-key")
    assert derived["provider"] == "gemini"
    assert derived["model"] == "gemini/gemini-embedding-001"
    assert derived["api_key"] == "gm-key"


def test_mistral_derives_mistral_embed_and_reuses_key():
    derived = derive_embedding_settings("mistral", None, "mi-key")
    assert derived["provider"] == "mistral"
    assert derived["model"] == "mistral/mistral-embed"
    assert derived["api_key"] == "mi-key"
    # mistral-embed is missing from litellm's registry — dims must be pinned.
    assert derived["dimensions"] == 1024


def test_ollama_uses_engine_defaults_and_llm_host():
    derived = derive_embedding_settings("ollama", "http://gpu-box:11434/v1", "ollama")
    assert derived["provider"] == "ollama"
    assert derived["model"] == "avr/sfr-embedding-mistral:latest"
    assert derived["endpoint"] == "http://gpu-box:11434/api/embed"
    assert derived["dimensions"] == 1024
    assert derived["huggingface_tokenizer"] == "Salesforce/SFR-Embedding-Mistral"


def test_ollama_without_llm_endpoint_falls_back_to_localhost():
    derived = derive_embedding_settings("ollama", None, None)
    assert derived["endpoint"] == "http://localhost:11434/api/embed"


def test_ollama_ignores_unparseable_llm_endpoint():
    derived = derive_embedding_settings("ollama", "not a url", None)
    assert derived["endpoint"] == "http://localhost:11434/api/embed"


def test_custom_derives_openai_compatible_against_llm_endpoint():
    derived = derive_embedding_settings("custom", "http://vllm:8000/v1", "vllm-key")
    assert derived["provider"] == "openai_compatible"
    assert derived["endpoint"] == "http://vllm:8000/v1"
    assert derived["api_key"] == "vllm-key"


@pytest.mark.parametrize("provider", ["anthropic", "llama_cpp"])
def test_providers_without_embedding_api_fail_fast(provider):
    with pytest.raises(EmbeddingProviderMismatchError) as exc_info:
        derive_embedding_settings(provider, None, "some-key")
    message = exc_info.value.message
    assert provider in message
    assert "EMBEDDING_PROVIDER=openai" in message
    assert "fastembed" in message


@pytest.mark.parametrize("provider", ["azure", "bedrock"])
def test_providers_needing_explicit_settings_fail_fast(provider):
    with pytest.raises(EmbeddingProviderMismatchError) as exc_info:
        derive_embedding_settings(provider, None, "some-key")
    message = exc_info.value.message
    assert provider in message
    assert "explicitly" in message


def test_unknown_provider_returns_none():
    assert derive_embedding_settings("some-future-provider", None, None) is None
    assert derive_embedding_settings("", None, None) is None
    assert derive_embedding_settings(None, None, None) is None


def test_provider_matching_is_case_insensitive():
    derived = derive_embedding_settings(" OpenAI ", None, "sk-test")
    assert derived["provider"] == "openai"


# --- wiring into get_embedding_engine ----------------------------------------


_EMBEDDING_ENV_VARS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_ENDPOINT",
    "EMBEDDING_API_KEY",
    "HUGGINGFACE_TOKENIZER",
    "MOCK_EMBEDDING",
)


def _clear_embedding_env(monkeypatch):
    for var in _EMBEDDING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _llm(provider, endpoint="", api_key="sk-test"):
    return types.SimpleNamespace(llm_provider=provider, llm_endpoint=endpoint, llm_api_key=api_key)


def _wire(monkeypatch, config, llm):
    """Patch config lookups and capture create_embedding_engine calls."""
    calls = []
    monkeypatch.setattr(engine_module, "get_embedding_context_config", lambda: config)
    monkeypatch.setattr(engine_module, "get_llm_context_config", lambda: llm)
    monkeypatch.setattr(
        engine_module, "create_embedding_engine", lambda *args: calls.append(args) or "engine"
    )
    return calls


def test_engine_derives_from_llm_provider_when_nothing_configured(monkeypatch):
    _clear_embedding_env(monkeypatch)
    config = EmbeddingConfig(_env_file=None)
    calls = _wire(monkeypatch, config, _llm("ollama", endpoint="http://gpu-box:11434/v1"))

    engine_module.get_embedding_engine()

    (provider, model, dimensions, _, endpoint, api_key, _, _, tokenizer, _, _) = calls[0]
    assert provider == "ollama"
    assert model == "avr/sfr-embedding-mistral:latest"
    assert dimensions == 1024
    assert endpoint == "http://gpu-box:11434/api/embed"
    assert api_key is None
    assert tokenizer == "Salesforce/SFR-Embedding-Mistral"


def test_engine_derivation_for_openai_matches_todays_defaults(monkeypatch):
    _clear_embedding_env(monkeypatch)
    config = EmbeddingConfig(_env_file=None)
    calls = _wire(monkeypatch, config, _llm("openai", api_key="sk-live"))

    engine_module.get_embedding_engine()

    (provider, model, dimensions, _, endpoint, api_key, _, _, _, _, _) = calls[0]
    assert provider == "openai"
    assert model == "openai/text-embedding-3-large"
    assert dimensions == 3072
    assert endpoint is None
    assert api_key == "sk-live"


def test_engine_raises_mismatch_for_anthropic_without_embedding_config(monkeypatch):
    _clear_embedding_env(monkeypatch)
    config = EmbeddingConfig(_env_file=None)
    _wire(monkeypatch, config, _llm("anthropic"))

    with pytest.raises(EmbeddingProviderMismatchError):
        engine_module.get_embedding_engine()


def test_explicit_embedding_env_var_wins_over_derivation(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    config = EmbeddingConfig(
        _env_file=None, embedding_provider="fastembed", embedding_model="BAAI/bge-small-en-v1.5"
    )
    calls = _wire(monkeypatch, config, _llm("anthropic"))

    engine_module.get_embedding_engine()  # must not raise

    assert calls[0][0] == "fastembed"
    assert calls[0][1] == "BAAI/bge-small-en-v1.5"


def test_programmatic_embedding_config_wins_over_derivation(monkeypatch):
    # cognee.config.set_embedding_provider() / explicit context configs mark
    # the field as set — no env var needed for "explicit config wins".
    _clear_embedding_env(monkeypatch)
    config = EmbeddingConfig(_env_file=None)
    config.embedding_provider = "fastembed"
    calls = _wire(monkeypatch, config, _llm("anthropic"))

    engine_module.get_embedding_engine()  # must not raise

    assert calls[0][0] == "fastembed"


def test_mock_embedding_keeps_default_path(monkeypatch):
    # MOCK_EMBEDDING test setups must keep working even with mismatched LLMs.
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    config = EmbeddingConfig(_env_file=None)
    calls = _wire(monkeypatch, config, _llm("anthropic"))

    engine_module.get_embedding_engine()  # must not raise

    assert calls[0][0] == config.embedding_provider


def test_explicit_tokenizer_survives_derivation(monkeypatch):
    _clear_embedding_env(monkeypatch)
    config = EmbeddingConfig(_env_file=None, huggingface_tokenizer="nomic-ai/nomic-embed-text-v1.5")
    calls = _wire(monkeypatch, config, _llm("ollama"))

    engine_module.get_embedding_engine()

    assert calls[0][8] == "nomic-ai/nomic-embed-text-v1.5"


# --- explicit-but-partial configs must keep origin/dev behavior ----------------


def test_embedding_api_key_alone_counts_as_explicit(monkeypatch):
    """A user setting only EMBEDDING_API_KEY relies on the documented OpenAI
    defaults; derivation must not reroute their embeddings (regression:
    custom/OpenRouter LLM + dedicated OpenAI embedding key)."""
    for var in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-real-openai-key")

    config = EmbeddingConfig()
    assert engine_module._embeddings_explicitly_configured(config) is True


def test_embedding_endpoint_alone_counts_as_explicit(monkeypatch):
    for var in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EMBEDDING_ENDPOINT", "https://embeddings.internal/v1")

    config = EmbeddingConfig()
    assert engine_module._embeddings_explicitly_configured(config) is True


def test_preflight_embedding_check_matches_engine(monkeypatch):
    """An anthropic-LLM user with EMBEDDING_API_KEY set worked on origin/dev
    (default OpenAI embeddings + their key) — preflight must not block them."""
    from cognee.cli import preflight

    for var in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-real-openai-key")

    assert preflight._embedding_env_configured() is True
    assert preflight._check_embeddings() is None
