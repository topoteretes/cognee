"""Tests for embedding-dimension auto-derivation in EmbeddingConfig."""

from unittest.mock import patch

import pytest

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
    for var in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS"):
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


# ---- Keyless default: untouched embeddings + no usable LLM key -> fastembed ----


def _llm(api_key):
    from types import SimpleNamespace

    return SimpleNamespace(
        llm_provider="openai", llm_api_key=api_key, llm_azure_use_managed_identity=False
    )


def _default_embeddings(monkeypatch, **overrides):
    _clear_embedding_env(monkeypatch)
    for var in ("EMBEDDING_API_KEY", "EMBEDDING_ENDPOINT", "EMBEDDING_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    return EmbeddingConfig(_env_file=None, **overrides)


def test_untouched_embeddings_without_llm_key_resolve_to_local_fastembed(monkeypatch):
    from cognee.infrastructure.databases.vector.embeddings.config import (
        DEFAULT_LOCAL_EMBEDDING_DIMENSIONS,
        DEFAULT_LOCAL_EMBEDDING_MODEL,
        resolve_embedding_defaults,
    )

    config = _default_embeddings(monkeypatch)
    resolved = resolve_embedding_defaults(config, _llm(api_key=None))

    assert resolved == (
        "fastembed",
        DEFAULT_LOCAL_EMBEDDING_MODEL,
        DEFAULT_LOCAL_EMBEDDING_DIMENSIONS,
    )
    # The config object itself is left as configured.
    assert config.embedding_provider == "openai"


def test_untouched_embeddings_with_llm_key_keep_the_openai_default(monkeypatch):
    from cognee.infrastructure.databases.vector.embeddings.config import (
        resolve_embedding_defaults,
    )

    config = _default_embeddings(monkeypatch)

    assert resolve_embedding_defaults(config, _llm(api_key="sk-test")) == (
        "openai",
        "openai/text-embedding-3-large",
        3072,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_provider": "ollama", "embedding_model": "nomic-embed-text"},
        {"embedding_model": "openai/text-embedding-3-small"},
        {"embedding_api_key": "sk-embed"},
        {"embedding_endpoint": "http://localhost:11434/api/embed"},
    ],
)
def test_any_configured_embedding_setting_is_honoured_without_llm_key(monkeypatch, overrides):
    from cognee.infrastructure.databases.vector.embeddings.config import (
        embeddings_untouched,
        resolve_embedding_defaults,
    )

    config = _default_embeddings(monkeypatch, **overrides)

    assert embeddings_untouched(config) is False
    assert resolve_embedding_defaults(config, _llm(api_key=None)) == (
        config.embedding_provider,
        config.embedding_model,
        config.embedding_dimensions,
    )


def test_engine_factory_uses_the_resolved_defaults(monkeypatch):
    import importlib

    # The package re-exports the function under the module's name, so the
    # module itself has to be imported by path to patch its collaborators.
    factory = importlib.import_module(
        "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine"
    )

    config = _default_embeddings(monkeypatch)
    captured = {}

    def fake_create(provider, model, dimensions, *args, **kwargs):
        captured.update(provider=provider, model=model, dimensions=dimensions)
        return object()

    monkeypatch.setattr(factory, "get_embedding_context_config", lambda: config)
    monkeypatch.setattr(factory, "get_llm_context_config", lambda: _llm(api_key=None))
    monkeypatch.setattr(factory, "create_embedding_engine", fake_create)

    factory.get_embedding_engine()

    assert captured == {
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
    }


# ---- EMBEDDING_API_BASE alias (SDK-539 / issue #4871) ----


def _fresh_config(monkeypatch, **env):
    """Build EmbeddingConfig from a controlled environment (no .env leakage
    for the two endpoint vars)."""
    from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig

    for var in ("EMBEDDING_ENDPOINT", "EMBEDDING_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    return EmbeddingConfig(_env_file=None)


def test_embedding_api_base_alone_populates_endpoint(monkeypatch):
    """The exact repro from issue #4871: API_BASE set, ENDPOINT not."""
    config = _fresh_config(monkeypatch, EMBEDDING_API_BASE="https://api.siliconflow.cn/v1")
    assert config.embedding_endpoint == "https://api.siliconflow.cn/v1"


def test_embedding_endpoint_wins_over_api_base(monkeypatch):
    config = _fresh_config(
        monkeypatch,
        EMBEDDING_ENDPOINT="https://endpoint.example/v1",
        EMBEDDING_API_BASE="https://api-base.example/v1",
    )
    assert config.embedding_endpoint == "https://endpoint.example/v1"


def test_embedding_endpoint_env_still_works_alone(monkeypatch):
    config = _fresh_config(monkeypatch, EMBEDDING_ENDPOINT="https://endpoint.example/v1")
    assert config.embedding_endpoint == "https://endpoint.example/v1"


def test_programmatic_field_name_construction_still_works(monkeypatch):
    from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig

    for var in ("EMBEDDING_ENDPOINT", "EMBEDDING_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    config = EmbeddingConfig(_env_file=None, embedding_endpoint="https://kwarg.example/v1")
    assert config.embedding_endpoint == "https://kwarg.example/v1"
