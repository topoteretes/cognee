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

    @pytest.mark.asyncio
    async def test_oversized_input_is_chunked_not_sent_whole(self, monkeypatch):
        """Inputs above the endpoint's input cap must be chunked, not sent whole.

        NVIDIA NIM rejects any request with more than 256 inputs with a 400
        ("input count 464 exceeds maximum allowed batch size") that no retry
        can fix — sending 464 texts in one call burned the full 128s retry
        budget and then failed. The engine must split first.
        """
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
        monkeypatch.delenv("EMBEDDING_MAX_INPUT_BATCH", raising=False)

        engine = self._make_engine(dimensions=8)

        def make_response(n):
            items = []
            for i in range(n):
                item = MagicMock()
                item.embedding = [float(i)] * 8
                items.append(item)
            resp = MagicMock()
            resp.data = items
            return resp

        calls = []

        async def fake_create(**kwargs):
            calls.append(len(kwargs["input"]))
            return make_response(len(kwargs["input"]))

        engine._client = MagicMock()
        engine._client.embeddings.create = AsyncMock(side_effect=fake_create)

        result = await engine.embed_text([f"text {i}" for i in range(464)])

        assert len(result) == 464
        # 464 inputs at the default 256 cap → chunks of 256 and 208
        assert calls == [256, 208]
        # Order preserved: first chunk's vectors come first
        assert result[0] == [0.0] * 8
        assert result[255] == [255.0] * 8
        assert result[256] == [0.0] * 8
        assert result[463] == [207.0] * 8

    @pytest.mark.asyncio
    async def test_batch_size_error_triggers_recursive_split(self, monkeypatch):
        """A 400 mentioning batch size must fall into the recursive split path."""
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
        monkeypatch.delenv("EMBEDDING_MAX_INPUT_BATCH", raising=False)

        engine = self._make_engine(dimensions=8)

        # Simulate an endpoint whose cap is lower than our default chunking
        call_count = {"n": 0}

        async def fake_create(**kwargs):
            call_count["n"] += 1
            if len(kwargs["input"]) > 4:
                raise RuntimeError(
                    "Error code: 400 - input count exceeds maximum allowed batch size"
                )
            items = []
            for i in range(len(kwargs["input"])):
                item = MagicMock()
                item.embedding = [1.0] * 8
                items.append(item)
            resp = MagicMock()
            resp.data = items
            return resp

        engine._client = MagicMock()
        engine._client.embeddings.create = AsyncMock(side_effect=fake_create)

        # Bypass pre-chunking (cap raised above input size) so only the
        # error-driven recursive split handles it.
        engine.max_input_batch = 100
        result = await engine.embed_text([f"t{i}" for i in range(8)])

        assert len(result) == 8
        assert all(v == [1.0] * 8 for v in result)
        # 8 > 4 → split to 4+4, both succeed
        assert call_count["n"] == 3
