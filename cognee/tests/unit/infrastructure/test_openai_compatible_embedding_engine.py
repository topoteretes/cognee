"""
Tests for OpenAICompatibleEmbeddingEngine.

Verifies that the engine:
- Returns mock embeddings when MOCK_EMBEDDING is set
- Calls the OpenAI SDK with encoding_format="float"
- Reports correct vector size and batch size
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenAICompatibleEmbeddingEngine:
    """Unit tests for OpenAICompatibleEmbeddingEngine."""

    def _make_engine(self, **kwargs):
        """Create an engine instance with defaults suitable for testing."""
        defaults = {
            "model": "test-model",
            "dimensions": 4096,
            "max_completion_tokens": 8191,
            "endpoint": "http://localhost:8099",
            "api_key": "test-key",
            "batch_size": 36,
        }
        defaults.update(kwargs)

        from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
            OpenAICompatibleEmbeddingEngine,
        )

        return OpenAICompatibleEmbeddingEngine(**defaults)

    @pytest.mark.asyncio
    async def test_mock_embedding(self, monkeypatch):
        """When MOCK_EMBEDDING=true, embed_text returns zero vectors of correct dimensions."""
        monkeypatch.setenv("MOCK_EMBEDDING", "true")
        engine = self._make_engine(dimensions=4096)
        result = await engine.embed_text(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 4096
        assert all(v == 0.0 for v in result[0])

    @pytest.mark.asyncio
    async def test_embed_text_calls_openai_with_encoding_format_float(self, monkeypatch):
        """embed_text must call OpenAI SDK with encoding_format='float'."""
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)

        engine = self._make_engine()

        # Build a mock response matching OpenAI SDK's CreateEmbeddingResponse
        mock_item = MagicMock()
        mock_item.embedding = [0.1] * 4096

        mock_response = MagicMock()
        mock_response.data = [mock_item]

        # Mock the AsyncOpenAI client's embeddings.create
        engine._client = MagicMock()
        engine._client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await engine.embed_text(["test text"])

        # Verify create was called with encoding_format="float"
        engine._client.embeddings.create.assert_called_once_with(
            model="test-model",
            input=["test text"],
            encoding_format="float",
        )

        assert len(result) == 1
        assert len(result[0]) == 4096

    def test_get_vector_size(self):
        """get_vector_size returns the configured dimensions."""
        engine = self._make_engine(dimensions=768)
        assert engine.get_vector_size() == 768

    def test_get_batch_size(self):
        """get_batch_size returns the configured batch size."""
        engine = self._make_engine(batch_size=50)
        assert engine.get_batch_size() == 50

    def test_max_completion_tokens_is_exposed(self):
        """The engine exposes max_completion_tokens for chunk sizing logic."""
        engine = self._make_engine(max_completion_tokens=2048)
        assert engine.max_completion_tokens == 2048

    def test_endpoint_normalization(self):
        """Endpoint without /v1 gets /v1 appended for the SDK base_url."""
        engine = self._make_engine(endpoint="http://localhost:8099")
        assert str(engine._client._base_url).rstrip("/").endswith("/v1")

        engine2 = self._make_engine(endpoint="http://localhost:8099/v1")
        assert str(engine2._client._base_url).rstrip("/").endswith("/v1")

        # Both should produce equivalent normalized URLs
        assert str(engine._client._base_url) == str(engine2._client._base_url)

    def test_endpoint_normalization_strips_embeddings_suffix(self):
        """Endpoint with /v1/embeddings should not produce /v1/embeddings/v1."""
        engine = self._make_engine(endpoint="http://localhost:8099/v1/embeddings")
        base_url = str(engine._client._base_url).rstrip("/")
        assert base_url.endswith("/v1")
        assert "/embeddings" not in base_url
