"""Unit tests for the zero-network provider-configuration preflight.

The check must catch exactly the two first-run traps observed in telemetry
(only-LLM configured with a non-OpenAI provider; only-embeddings configured
with the LLM key requirement unmet) and stay silent for every consistent
pairing — including the keyless providers (bedrock, llama_cpp, azure with
managed identity) and keyless embedders (fastembed, local endpoints).

Configs are duck-typed here (SimpleNamespace) so the tests cannot be
polluted by a developer's real .env, which BaseSettings would load.
"""

from types import SimpleNamespace

import pytest

from cognee.modules.preflight import config_preflight
from cognee.modules.preflight import (
    ProviderConfigMismatchError,
    check_provider_config,
    reset_preflight_state,
    validate_provider_config,
)

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-large"


def llm(provider="openai", api_key: "str | None" = "sk-test", managed_identity=False):
    return SimpleNamespace(
        llm_provider=provider,
        llm_api_key=api_key,
        llm_azure_use_managed_identity=managed_identity,
    )


def embeddings(
    provider="openai",
    model=DEFAULT_EMBEDDING_MODEL,
    api_key=None,
    endpoint=None,
):
    return SimpleNamespace(
        embedding_provider=provider,
        embedding_model=model,
        embedding_api_key=api_key,
        embedding_endpoint=endpoint,
    )


class TestOnlyLLMConfiguredTrap:
    def test_non_openai_llm_with_default_embeddings_is_flagged(self):
        problems = check_provider_config(llm(provider="anthropic"), embeddings())
        assert len(problems) == 1
        assert "EMBEDDING_PROVIDER" in problems[0]
        assert "anthropic" in problems[0]

    def test_custom_provider_with_default_embeddings_is_flagged(self):
        problems = check_provider_config(llm(provider="custom"), embeddings())
        assert len(problems) == 1
        assert "no API key at all" in problems[0]

    def test_openai_llm_with_default_embeddings_is_fine(self):
        # The documented convenience: an OpenAI LLM_API_KEY is reused for the
        # default OpenAI embedder by design.
        assert check_provider_config(llm(provider="openai"), embeddings()) == []

    def test_non_openai_llm_with_configured_embeddings_is_fine(self):
        problems = check_provider_config(
            llm(provider="anthropic"),
            embeddings(provider="fastembed", model="BAAI/bge-small-en-v1.5"),
        )
        assert problems == []

    def test_non_openai_llm_with_openai_embedding_key_is_fine(self):
        problems = check_provider_config(
            llm(provider="gemini"), embeddings(api_key="sk-openai-key")
        )
        assert problems == []

    def test_local_embedding_endpoint_is_fine(self):
        problems = check_provider_config(
            llm(provider="ollama"),
            embeddings(model=DEFAULT_EMBEDDING_MODEL, endpoint="http://localhost:11434/api/embed"),
        )
        assert problems == []


class TestOnlyEmbeddingsConfiguredTrap:
    def test_configured_embeddings_without_llm_key_is_flagged(self):
        problems = check_provider_config(
            llm(provider="openai", api_key=None),
            embeddings(provider="fastembed", model="BAAI/bge-small-en-v1.5"),
        )
        assert len(problems) == 1
        assert "LLM_API_KEY" in problems[0]

    def test_whitespace_llm_key_counts_as_missing(self):
        problems = check_provider_config(
            llm(api_key="   "),
            embeddings(api_key="sk-embed"),
        )
        assert len(problems) == 1
        assert "LLM_API_KEY" in problems[0]

    def test_bedrock_needs_no_llm_key(self):
        problems = check_provider_config(
            llm(provider="bedrock", api_key=None),
            embeddings(provider="fastembed", model="BAAI/bge-small-en-v1.5"),
        )
        assert problems == []

    def test_llama_cpp_needs_no_llm_key(self):
        problems = check_provider_config(
            llm(provider="llama_cpp", api_key=None),
            embeddings(api_key="sk-embed"),
        )
        assert problems == []

    def test_azure_managed_identity_needs_no_llm_key(self):
        problems = check_provider_config(
            llm(provider="azure", api_key=None, managed_identity=True),
            embeddings(api_key="sk-embed"),
        )
        assert problems == []


class TestFullyConfiguredAndUnconfigured:
    def test_both_fully_configured_is_fine(self):
        problems = check_provider_config(
            llm(provider="anthropic", api_key="sk-ant"),
            embeddings(provider="ollama", model="nomic-embed-text", endpoint="http://localhost"),
        )
        assert problems == []

    def test_nothing_configured_is_not_a_preflight_problem(self):
        # No key at all on pure defaults is the existing LLMAPIKeyNotSetError
        # path — the preflight only owns the *inconsistency* traps.
        problems = check_provider_config(llm(api_key=None), embeddings())
        assert problems == []


class TestValidateProviderConfig:
    @pytest.fixture(autouse=True)
    def fresh_state(self):
        reset_preflight_state()
        yield
        reset_preflight_state()

    def test_raises_on_problems(self, monkeypatch):
        monkeypatch.delenv("COGNEE_SKIP_PREFLIGHT", raising=False)
        monkeypatch.delenv("COGNEE_SKIP_CONNECTION_TEST", raising=False)
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
        monkeypatch.setattr(
            config_preflight, "check_provider_config", lambda *a, **k: ["problem one"]
        )
        with pytest.raises(ProviderConfigMismatchError, match="problem one"):
            validate_provider_config()
        # Failure is not cached: a second call re-validates.
        with pytest.raises(ProviderConfigMismatchError):
            validate_provider_config()

    def test_success_is_cached_once_per_process(self, monkeypatch):
        monkeypatch.delenv("COGNEE_SKIP_PREFLIGHT", raising=False)
        monkeypatch.delenv("COGNEE_SKIP_CONNECTION_TEST", raising=False)
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
        calls = []
        monkeypatch.setattr(
            config_preflight, "check_provider_config", lambda *a, **k: calls.append(1) or []
        )
        validate_provider_config()
        validate_provider_config()
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "env_var", ["COGNEE_SKIP_PREFLIGHT", "COGNEE_SKIP_CONNECTION_TEST", "MOCK_EMBEDDING"]
    )
    def test_skip_env_vars_disable_the_check(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "true")
        monkeypatch.setattr(
            config_preflight,
            "check_provider_config",
            lambda *a, **k: pytest.fail("check ran despite skip env var"),
        )
        validate_provider_config()
