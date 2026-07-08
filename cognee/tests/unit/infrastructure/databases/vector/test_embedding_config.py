"""Tests for embedding-dimension auto-derivation in EmbeddingConfig."""

from unittest.mock import patch

import pytest

from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.vector.embeddings.config import (
    EmbeddingConfig,
    _resolve_embedding_dimensions,
)
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    _get_embedding_api_key,
    _validate_embedding_engine_config,
)


def test_resolve_openai_text_embedding_3_large():
    # Default cognee config — must keep mapping to 3072 for back-compat.
    assert _resolve_embedding_dimensions("openai", "openai/text-embedding-3-large") == 3072


def test_resolve_openai_text_embedding_3_small():
    # Bare model names are accepted too.
    assert _resolve_embedding_dimensions("openai", "text-embedding-3-small") == 1536


def test_resolve_unknown_model_returns_none():
    # Models not in any registry should signal "unknown" rather than guessing.
    dim = _resolve_embedding_dimensions("openai", "no-such-embedder-xyzzy")
    assert dim is None


def test_resolve_returns_none_for_missing_inputs():
    assert _resolve_embedding_dimensions(None, None) is None
    assert _resolve_embedding_dimensions("openai", None) is None
    assert _resolve_embedding_dimensions("", "model") is None


def test_resolve_fastembed_uses_registry():
    # Fake the fastembed import to verify we read both `dim` and `embed_dim`
    # variants without requiring fastembed to be installed.
    import sys
    import types

    fake_module = types.ModuleType("fastembed")

    class _FakeTextEmbedding:
        @staticmethod
        def list_supported_models():
            return [
                {"model": "BAAI/bge-small-en-v1.5", "dim": 384},
                {"model": "BAAI/bge-large-en-v1.5", "embed_dim": 1024},
            ]

    fake_module.TextEmbedding = _FakeTextEmbedding

    with patch.dict(sys.modules, {"fastembed": fake_module}):
        assert _resolve_embedding_dimensions("fastembed", "BAAI/bge-small-en-v1.5") == 384
        assert _resolve_embedding_dimensions("fastembed", "BAAI/bge-large-en-v1.5") == 1024


def _clear_embedding_env(monkeypatch):
    # CI workflows set EMBEDDING_* env vars (see .github/workflows/basic_tests.yml),
    # and pydantic-settings reads them at construction time — bypassing the
    # class defaults and the auto-resolve path we want to test here.
    for var in (
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_MAX_COMPLETION_TOKENS",
        "EMBEDDING_MAX_TOKENS",
    ):
        monkeypatch.delenv(var, raising=False)


def test_config_auto_resolves_when_dimensions_unset(monkeypatch):
    # Default config should still produce 3072 (the OpenAI default model).
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(_env_file=None)
    assert cfg.embedding_dimensions == 3072


def test_config_honors_explicit_dimensions(monkeypatch):
    # Explicit override must win over auto-resolution.
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(
        _env_file=None,
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimensions=384,
    )
    assert cfg.embedding_dimensions == 384


def test_config_fails_when_unresolvable_dimensions_are_unset(monkeypatch):
    _clear_embedding_env(monkeypatch)

    with pytest.raises(CogneeConfigurationError, match="Set EMBEDDING_DIMENSIONS"):
        EmbeddingConfig(
            _env_file=None,
            embedding_provider="openai",
            embedding_model="totally-fake-embedder-zzz",
            embedding_dimensions=None,
        )


def test_config_allows_unknown_model_with_explicit_dimensions(monkeypatch):
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(
        _env_file=None,
        embedding_provider="custom",
        embedding_model="my-private-embedding-model",
        embedding_dimensions=1536,
    )

    assert cfg.embedding_dimensions == 1536


def test_config_accepts_embedding_max_tokens_alias(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_MAX_TOKENS", "4096")

    cfg = EmbeddingConfig(_env_file=None)

    assert cfg.embedding_max_completion_tokens == 4096


def test_config_defaults_embedding_batch_size(monkeypatch):
    _clear_embedding_env(monkeypatch)

    cfg = EmbeddingConfig(_env_file=None)

    assert cfg.embedding_batch_size == 36


def test_engine_validation_rejects_silent_openai_embedding_default_for_non_openai_llm(
    monkeypatch,
):
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(_env_file=None)

    with pytest.raises(CogneeConfigurationError, match="default OpenAI embeddings"):
        _validate_embedding_engine_config(
            cfg.embedding_provider,
            cfg.embedding_model,
            cfg.embedding_dimensions,
            "anthropic",
            cfg.model_fields_set,
        )


def test_engine_validation_allows_explicit_openai_embeddings_with_non_openai_llm(
    monkeypatch,
):
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(
        _env_file=None,
        embedding_model="openai/text-embedding-3-large",
        embedding_dimensions=3072,
    )

    _validate_embedding_engine_config(
        cfg.embedding_provider,
        cfg.embedding_model,
        cfg.embedding_dimensions,
        "anthropic",
        cfg.model_fields_set,
    )


def test_engine_key_fallback_reuses_key_for_same_provider():
    assert (
        _get_embedding_api_key(None, "same-provider-key", "openai", "openai")
        == "same-provider-key"
    )


def test_engine_key_fallback_does_not_reuse_mismatched_llm_key():
    assert _get_embedding_api_key(None, "anthropic-key", "openai", "anthropic") is None


def test_engine_key_fallback_prefers_explicit_embedding_key():
    assert (
        _get_embedding_api_key("embedding-key", "llm-key", "openai", "anthropic")
        == "embedding-key"
    )
