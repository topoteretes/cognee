"""
Tests for OpenAICompatibleEmbeddingEngine.

Verifies that the engine:
- Returns mock embeddings when MOCK_EMBEDDING is set
- Requests float by default and base64 only when EMBEDDING_ENCODING_FORMAT opts in
- Reports correct vector size and batch size
"""

import base64
import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncOpenAI
from tenacity import stop_after_attempt


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
        """embed_text must call OpenAI SDK with encoding_format='float' by default."""
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
            encoding_format="float",
        )

        assert len(result) == 1
        assert len(result[0]) == 4096

    @pytest.mark.asyncio
    async def test_opted_in_base64_is_requested_and_decoded(self, monkeypatch):
        """With encoding_format='base64' the wire asks for base64; floats come back."""
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

        engine = self._make_engine(dimensions=4, encoding_format="base64")
        engine._client = AsyncOpenAI(
            api_key="test-key",
            base_url="http://localhost:8099",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        result = await engine.embed_text(["a", "b"])

        assert sent_bodies[0]["encoding_format"] == "base64"
        assert result == [float32_vector, float32_vector]
        assert all(isinstance(v, float) for v in result[0])

    @pytest.mark.asyncio
    async def test_float_list_on_base64_fails_without_retry(self, monkeypatch):
        """A server that answers base64 with a float list is a terminal configuration error."""
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
        from cognee.infrastructure.databases.exceptions import EmbeddingEncodingError

        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 1.0, 1.0]}
                    ],
                    "model": "test-model",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )

        engine = self._make_engine(dimensions=4, encoding_format="base64")
        engine._client = AsyncOpenAI(
            api_key="test-key",
            base_url="http://localhost:8099",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(EmbeddingEncodingError, match="EMBEDDING_ENCODING_FORMAT"):
            await engine.embed_text(["a"])

        assert len(calls) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "body", "hinted"),
        [
            (400, {"error": {"message": "unsupported encoding_format"}}, True),
            (422, {"error": {"message": "encoding_format: Input should be 'float'"}}, True),
            (400, {"error": {"message": "model 'nope' not found"}}, False),
            # FastAPI's own 422 locates the refused field...
            (
                422,
                {
                    "detail": [
                        {
                            "type": "literal_error",
                            "loc": ["body", "encoding_format"],
                            "msg": "Input should be 'float'",
                            "input": "base64",
                        }
                    ]
                },
                True,
            ),
            # A framework 400 whose "error" member is a bare string keeps the reason in
            # "message"; the openai SDK's body holds only "Bad Request".
            (
                400,
                {
                    "statusCode": 400,
                    "error": "Bad Request",
                    "message": "body/encoding_format must be equal to one of the allowed values",
                },
                True,
            ),
            # FastAPI echoes the whole request when the error is about something else.
            (
                422,
                {
                    "detail": [
                        {
                            "type": "missing",
                            "loc": ["body", "input_type"],
                            "msg": "Field required",
                            "input": {"input": ["a"], "encoding_format": "base64"},
                        }
                    ]
                },
                False,
            ),
        ],
    )
    async def test_rejection_naming_the_encoding_is_terminal(
        self, monkeypatch, status, body, hinted
    ):
        """A 400/422 about base64 raises EmbeddingEncodingError on the first attempt; an
        unrelated 400 keeps the engine's generic wrapper and retry policy."""
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
        from cognee.infrastructure.databases.exceptions import (
            EmbeddingEncodingError,
            EmbeddingException,
        )

        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(status, json=body)

        engine = self._make_engine(dimensions=4, encoding_format="base64")
        engine._client = AsyncOpenAI(
            api_key="test-key",
            base_url="http://localhost:8099",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        # The generic wrapper is retried with backoff; cap it so the unrelated case
        # finishes. The encoding error must stop on its own, before any retry.
        engine.embed_text = engine.embed_text.retry_with(stop=stop_after_attempt(2)).__get__(engine)

        if hinted:
            with pytest.raises(EmbeddingEncodingError, match="EMBEDDING_ENCODING_FORMAT"):
                await engine.embed_text(["a"])
            assert len(calls) == 1
        else:
            with pytest.raises(EmbeddingException):
                await engine.embed_text(["a"])
            assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_a_float_engine_never_blames_base64(self, monkeypatch):
        """Only an engine that asked for base64 turns a refusal of it into
        EmbeddingEncodingError; with float requested the setting is not the cause."""
        monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
        from cognee.infrastructure.databases.exceptions import (
            EmbeddingEncodingError,
            EmbeddingException,
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "unsupported encoding_format"}})

        engine = self._make_engine(dimensions=4)
        engine._client = AsyncOpenAI(
            api_key="test-key",
            base_url="http://localhost:8099",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        engine.embed_text = engine.embed_text.retry_with(stop=stop_after_attempt(1)).__get__(engine)

        with pytest.raises(EmbeddingException) as info:
            await engine.embed_text(["a"])
        assert not isinstance(info.value, EmbeddingEncodingError)

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
