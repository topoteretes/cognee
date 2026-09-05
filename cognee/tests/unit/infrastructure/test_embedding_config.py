"""
Tests for EmbeddingConfig.model_post_init and EmbeddingConfig.to_dict.

Covers three defects fixed in issue #3902:

  1. Dead if/elif: both branches set batch_size = 36, the provider check was
     pointless dead code.

  2. AttributeError when embedding_provider is None: model_post_init called
     self.embedding_provider.lower() without a None guard. Because the field is
     Optional[str], passing provider=None crashed construction entirely.

  3. to_dict() silently dropped embedding_batch_size. Any code that serialized
     the config (for caching, logging, or restoring config state) lost the
     batch size silently.

All tests are pure-Python. They mock _resolve_embedding_dimensions so that
litellm / fastembed do not need to be installed for the suite to pass.
"""

from unittest.mock import patch

import pytest

_RESOLVE = "cognee.infrastructure.databases.vector.embeddings.config._resolve_embedding_dimensions"


def _make_config(**kwargs):
    from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
    return EmbeddingConfig(**kwargs)


class TestBatchSizeDefault:
    """embedding_batch_size defaults to 36 for every provider when not set."""

    def test_openai_provider_gets_default_batch_size(self):
        with patch(_RESOLVE, return_value=1536):
            cfg = _make_config(embedding_provider="openai", embedding_batch_size=None)
        assert cfg.embedding_batch_size == 36

    def test_non_openai_provider_also_gets_default_batch_size(self):
        # Before the fix, the elif branch still set 36. The provider check was dead.
        with patch(_RESOLVE, return_value=768):
            cfg = _make_config(embedding_provider="fastembed", embedding_batch_size=None)
        assert cfg.embedding_batch_size == 36

    def test_explicit_batch_size_is_not_overwritten(self):
        with patch(_RESOLVE, return_value=1536):
            cfg = _make_config(embedding_provider="openai", embedding_batch_size=64)
        assert cfg.embedding_batch_size == 64

    def test_explicit_batch_size_preserved_for_non_openai(self):
        with patch(_RESOLVE, return_value=768):
            cfg = _make_config(embedding_provider="cohere", embedding_batch_size=128)
        assert cfg.embedding_batch_size == 128


class TestNoneProvider:
    """provider=None must not raise AttributeError during construction."""

    def test_none_provider_does_not_raise(self):
        # Before the fix: self.embedding_provider.lower() raised
        # AttributeError: 'NoneType' object has no attribute 'lower'
        try:
            cfg = _make_config(embedding_provider=None, embedding_batch_size=None)
        except AttributeError as exc:
            pytest.fail(
                f"EmbeddingConfig raised AttributeError with provider=None: {exc}"
            )
        assert cfg.embedding_batch_size == 36

    def test_none_provider_explicit_batch_size_preserved(self):
        cfg = _make_config(embedding_provider=None, embedding_batch_size=48)
        assert cfg.embedding_batch_size == 48


class TestToDictIncludesBatchSize:
    """to_dict() must include embedding_batch_size."""

    def test_key_is_present(self):
        with patch(_RESOLVE, return_value=1536):
            cfg = _make_config()
        assert "embedding_batch_size" in cfg.to_dict()

    def test_value_matches_instance(self):
        with patch(_RESOLVE, return_value=1536):
            cfg = _make_config(embedding_batch_size=72)
        assert cfg.to_dict()["embedding_batch_size"] == 72

    def test_default_value_is_serialized(self):
        with patch(_RESOLVE, return_value=1536):
            cfg = _make_config(embedding_batch_size=None)
        assert cfg.to_dict()["embedding_batch_size"] == 36

    def test_no_existing_keys_removed(self):
        expected = {
            "embedding_provider",
            "embedding_model",
            "embedding_dimensions",
            "embedding_endpoint",
            "embedding_api_key",
            "embedding_api_version",
            "embedding_max_completion_tokens",
            "embedding_batch_size",
            "huggingface_tokenizer",
        }
        with patch(_RESOLVE, return_value=1536):
            cfg = _make_config()
        assert set(cfg.to_dict().keys()) == expected

    def test_round_trip_preserves_batch_size(self):
        # Simulate a cache-restore: serialize then reconstruct from the dict.
        with patch(_RESOLVE, return_value=1536):
            original = _make_config(embedding_batch_size=99)
        data = {k: v for k, v in original.to_dict().items() if v is not None}
        with patch(_RESOLVE, return_value=1536):
            restored = _make_config(**data)
        assert restored.embedding_batch_size == 99
