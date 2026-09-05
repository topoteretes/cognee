"""Tests for EmbeddingConfig fixes - Issue #3902

Verifies:
1. Dead conditional removed: batch_size default doesn't depend on provider
2. None provider guard: no AttributeError when embedding_provider is None
3. to_dict() includes embedding_batch_size field
"""
import pytest
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig


class TestEmbeddingConfigDeadCodeRemoval:
    """Fix #1: Merged dead conditional - batch_size default is provider-independent."""

    def test_default_batch_size_with_openai(self):
        config = EmbeddingConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
        )
        assert config.embedding_batch_size == 36

    def test_default_batch_size_with_non_openai_provider(self):
        config = EmbeddingConfig(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            embedding_dimensions=768,
        )
        assert config.embedding_batch_size == 36

    def test_explicit_batch_size_preserved(self):
        config = EmbeddingConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            embedding_batch_size=64,
        )
        assert config.embedding_batch_size == 64


class TestEmbeddingConfigNoneProviderGuard:
    """Fix #2: provider=None no longer causes AttributeError."""

    def test_none_provider_no_crash(self):
        config = EmbeddingConfig(
            embedding_provider=None,
            embedding_model="some-model",
            embedding_dimensions=768,
        )
        assert config.embedding_provider is None
        assert config.embedding_batch_size == 36

    def test_none_provider_with_explicit_batch_size(self):
        config = EmbeddingConfig(
            embedding_provider=None,
            embedding_model="some-model",
            embedding_dimensions=768,
            embedding_batch_size=128,
        )
        assert config.embedding_batch_size == 128


class TestEmbeddingConfigToDict:
    """Fix #3: to_dict() now includes embedding_batch_size."""

    def test_to_dict_contains_batch_size(self):
        config = EmbeddingConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            embedding_batch_size=64,
        )
        result = config.to_dict()
        assert "embedding_batch_size" in result
        assert result["embedding_batch_size"] == 64

    def test_to_dict_default_batch_size(self):
        config = EmbeddingConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
        )
        result = config.to_dict()
        assert "embedding_batch_size" in result
        assert result["embedding_batch_size"] == 36

    def test_to_dict_roundtrip_all_fields(self):
        config = EmbeddingConfig(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            embedding_dimensions=768,
            embedding_batch_size=100,
        )
        result = config.to_dict()
        assert result["embedding_provider"] == "ollama"
        assert result["embedding_model"] == "nomic-embed-text"
        assert result["embedding_dimensions"] == 768
        assert result["embedding_batch_size"] == 100
        assert result["embedding_max_completion_tokens"] == 8191
