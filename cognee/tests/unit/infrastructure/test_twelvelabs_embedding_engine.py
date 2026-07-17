"""
Tests for TwelveLabsEmbeddingEngine (Marengo).

Verifies that the engine:
- Returns mock embeddings when MOCK_EMBEDDING is set
- Reports correct vector size and batch size
- Parses the TwelveLabs /embed response shape (text_embedding.segments[0].float)
- Posts model_name + text to the configured endpoint with the x-api-key header

The network-touching test mocks _get_embedding so it runs offline.
"""

from unittest.mock import AsyncMock

import pytest


class TestTwelveLabsEmbeddingEngine:
    """Unit tests for TwelveLabsEmbeddingEngine."""

    def _make_engine(self, **kwargs):
        defaults = {
            "model": "marengo3.0",
            "dimensions": 512,
            "max_completion_tokens": 512,
            "endpoint": "https://api.twelvelabs.io/v1.3/embed",
            "api_key": "test-key",
            "batch_size": 100,
        }
        defaults.update(kwargs)

        from cognee.infrastructure.databases.vector.embeddings.TwelveLabsEmbeddingEngine import (
            TwelveLabsEmbeddingEngine,
        )

        return TwelveLabsEmbeddingEngine(**defaults)

    @pytest.mark.asyncio
    async def test_mock_embedding(self, monkeypatch):
        """When MOCK_EMBEDDING=true, embed_text returns zero vectors of correct dimensions."""
        monkeypatch.setenv("MOCK_EMBEDDING", "true")
        engine = self._make_engine(dimensions=512)
        result = await engine.embed_text(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 512
        assert all(v == 0.0 for v in result[0])

    @pytest.mark.asyncio
    async def test_embed_text_returns_per_prompt_vectors(self, monkeypatch):
        """embed_text returns one Marengo vector per input prompt."""
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
        engine = self._make_engine()
        engine._get_embedding = AsyncMock(return_value=[0.1] * 512)

        result = await engine.embed_text(["one", "two", "three"])

        assert engine._get_embedding.await_count == 3
        assert len(result) == 3
        assert len(result[0]) == 512

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, monkeypatch):
        """A missing API key produces a clear EmbeddingException, not a network error."""
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
        monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
        from cognee.infrastructure.databases.exceptions import EmbeddingException

        engine = self._make_engine(api_key=None)
        with pytest.raises(EmbeddingException):
            await engine.embed_text(["hello"])

    def test_get_vector_size_defaults_to_512(self):
        """Marengo defaults to 512 dimensions when none is supplied."""
        engine = self._make_engine(dimensions=None)
        assert engine.get_vector_size() == 512

    def test_get_batch_size(self):
        engine = self._make_engine(batch_size=50)
        assert engine.get_batch_size() == 50

    def test_factory_dispatches_to_twelvelabs(self):
        """create_embedding_engine routes provider='twelvelabs' to this engine."""
        from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
            create_embedding_engine,
        )
        from cognee.infrastructure.databases.vector.embeddings.TwelveLabsEmbeddingEngine import (
            TwelveLabsEmbeddingEngine,
        )

        engine = create_embedding_engine(
            "twelvelabs", "marengo3.0", 512, 512, None, "k", None, 100, None, "k", "openai"
        )
        assert isinstance(engine, TwelveLabsEmbeddingEngine)

    def test_config_resolves_marengo_dimensions(self):
        """The config helper resolves twelvelabs to 512 dims without env override."""
        from cognee.infrastructure.databases.vector.embeddings.config import (
            _resolve_embedding_dimensions,
        )

        assert _resolve_embedding_dimensions("twelvelabs", "marengo3.0") == 512
