"""Tests for embedding-dimension auto-derivation in EmbeddingConfig."""

from unittest.mock import patch

from cognee.infrastructure.databases.vector.embeddings.config import (
    EmbeddingConfig,
    _resolve_embedding_dimensions,
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
        "EMBEDDING_BATCH_SIZE",
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


def test_config_falls_back_when_unresolvable(monkeypatch):
    # Unknown model + unset dimensions falls back to 3072 with a warning,
    # so existing setups don't crash at import time.
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(
        _env_file=None,
        embedding_provider="openai",
        embedding_model="totally-fake-embedder-zzz",
        embedding_dimensions=None,
    )
    assert cfg.embedding_dimensions == 3072


def test_batch_size_defaults_to_36(monkeypatch):
    # Unset batch size resolves to the single provider-agnostic default.
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(_env_file=None)
    assert cfg.embedding_batch_size == 36


def test_batch_size_defaults_to_36_for_non_openai(monkeypatch):
    # The old dead if/elif made this look provider-specific; it never was.
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(
        _env_file=None,
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimensions=384,
    )
    assert cfg.embedding_batch_size == 36


def test_batch_size_honors_explicit_value(monkeypatch):
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(_env_file=None, embedding_batch_size=8)
    assert cfg.embedding_batch_size == 8


def test_none_provider_does_not_crash(monkeypatch):
    # Regression: model_post_init used to call embedding_provider.lower()
    # unconditionally, raising AttributeError when the provider was None.
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(
        _env_file=None,
        embedding_provider=None,
        embedding_dimensions=384,
    )
    assert cfg.embedding_batch_size == 36


def test_to_dict_includes_batch_size(monkeypatch):
    # Regression: to_dict() dropped embedding_batch_size, silently losing it
    # on a round-trip.
    _clear_embedding_env(monkeypatch)
    cfg = EmbeddingConfig(_env_file=None)
    result = cfg.to_dict()
    assert "embedding_batch_size" in result
    assert result["embedding_batch_size"] == cfg.embedding_batch_size
