"""
Tests for OpenAICompatibleEmbeddingEngine.

Verifies that the engine:
- Returns mock embeddings when MOCK_EMBEDDING is set
- Leaves encoding_format to the OpenAI SDK, which requests and decodes base64
- Reports correct vector size and batch size
"""

import base64
import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncOpenAI


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
    async def test_embed_text_leaves_encoding_format_to_the_sdk(self, monkeypatch):
        """embed_text must not pin encoding_format, so the SDK's base64 default applies."""
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

        engine._client.embeddings.create.assert_called_once_with(
            model="test-model",
            input=["test text"],
        )

        assert len(result) == 1
        assert len(result[0]) == 4096

    @pytest.mark.asyncio
    async def test_embed_text_requests_and_decodes_base64(self, monkeypatch):
        """Over the wire the SDK asks for base64; the engine returns the float32 values."""
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)

        vector = [0.0123, -0.5, 0.25, 1e-7]
        float32_vector = [struct.unpack("<f", struct.pack("<f", v))[0] for v in vector]
        sent_bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            sent_bodies.append(body)
            encoded = base64.b64encode(struct.pack("<4f", *vector)).decode()
            data = [
                {"object": "embedding", "index": i, "embedding": encoded}
                for i in range(len(body["input"]))
            ]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": data,
                    "model": body["model"],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )

        engine = self._make_engine(dimensions=4)
        engine._client = AsyncOpenAI(
            api_key="test-key",
            base_url="http://localhost:8099",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        result = await engine.embed_text(["a", "b"])

        assert sent_bodies[0]["encoding_format"] == "base64"
        assert result == [float32_vector, float32_vector]
        assert all(isinstance(v, float) for v in result[0])

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
