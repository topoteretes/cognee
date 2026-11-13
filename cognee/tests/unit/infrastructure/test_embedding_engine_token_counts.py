"""Each engine's embed_text() reports the input tokens of the requests that actually
succeeded, through its split, pool and per-prompt paths."""

from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)


class _WordTokenizer:
    def count_tokens(self, text):
        return len(text.split())


def _litellm_engine():
    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=_WordTokenizer()):
        return LiteLLMEmbeddingEngine(model="test-model", dimensions=2)


def _too_long():
    return litellm.exceptions.BadRequestError(
        message="maximum input length is 8192 tokens", model="test-model", llm_provider="openai"
    )


@pytest.mark.asyncio
async def test_litellm_split_counts_only_the_requests_that_succeeded():
    async def fake_aembedding(*, input, **kwargs):
        if len(input) > 1:
            raise _too_long()
        return SimpleNamespace(
            data=[{"embedding": [1.0, 0.0]}], usage=SimpleNamespace(prompt_tokens=10)
        )

    with patch("litellm.aembedding", side_effect=fake_aembedding):
        result = await _litellm_engine().embed_text(["a", "b", "c"])

    assert len(result) == 3
    assert result.tokens == 30


@pytest.mark.asyncio
async def test_litellm_pool_counts_both_overlapping_parts():
    long_text = "x" * 12

    async def fake_aembedding(*, input, **kwargs):
        if input == [long_text]:
            raise _too_long()
        return SimpleNamespace(
            data=[{"embedding": [1.0, 0.0]}], usage=SimpleNamespace(prompt_tokens=7)
        )

    with patch("litellm.aembedding", side_effect=fake_aembedding):
        result = await _litellm_engine().embed_text([long_text])

    assert len(result) == 1
    assert result.tokens == 14


@pytest.mark.asyncio
async def test_openai_compatible_uses_the_reported_usage():
    from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
        OpenAICompatibleEmbeddingEngine,
    )

    with patch.object(
        OpenAICompatibleEmbeddingEngine, "get_tokenizer", return_value=_WordTokenizer()
    ):
        engine = OpenAICompatibleEmbeddingEngine(
            model="test-model", dimensions=2, endpoint="http://localhost:1/v1", api_key="x"
        )

    async def fake_create(*, input, **kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input],
            usage=SimpleNamespace(prompt_tokens=9),
        )

    engine._client.embeddings.create = fake_create
    result = await engine.embed_text(["a", "b"])

    assert len(result) == 2
    assert result.tokens == 9


class _FakeAiohttpResponse:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


@pytest.mark.asyncio
async def test_ollama_sums_the_per_prompt_counts(monkeypatch):
    import aiohttp

    from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
        OllamaEmbeddingEngine,
    )

    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    with patch.object(OllamaEmbeddingEngine, "get_tokenizer", return_value=_WordTokenizer()):
        engine = OllamaEmbeddingEngine(model="nomic-embed-text", dimensions=2)

    def fake_post(self, url, *, json, **kwargs):
        return _FakeAiohttpResponse({"embeddings": [[1.0, 0.0]], "prompt_eval_count": 4})

    monkeypatch.setattr(aiohttp.ClientSession, "post", fake_post)
    result = await engine.embed_text(["a", "b", "c"])

    assert len(result) == 3
    assert result.tokens == 12


FASTEMBED = "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine"


@pytest.mark.asyncio
async def test_fastembed_counts_with_its_own_tokenizer(monkeypatch):
    import numpy as np

    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    with (
        patch(f"{FASTEMBED}.FastembedEmbeddingEngine.get_tokenizer", return_value=_WordTokenizer()),
        patch(f"{FASTEMBED}.TextEmbedding") as text_embedding,
    ):
        text_embedding.return_value.embed.side_effect = lambda inputs, **kwargs: iter(
            [np.array([1.0, 0.0]) for _ in inputs]
        )
        from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
            FastembedEmbeddingEngine,
        )

        engine = FastembedEmbeddingEngine(model="test-model", dimensions=2)
        result = await engine.embed_text(["two words", "one"])

    assert len(result) == 2
    assert result.tokens == 3


@pytest.mark.asyncio
async def test_mock_embeddings_consume_no_tokens(monkeypatch):
    monkeypatch.setenv("MOCK_EMBEDDING", "true")

    result = await _litellm_engine().embed_text(["a", "b"])

    assert len(result) == 2
    assert result.tokens == 0


@pytest.mark.asyncio
async def test_embed_text_still_returns_a_plain_list_of_vectors(monkeypatch):
    """Direct callers of an engine keep the pre-token-counting contract: a list of
    vectors, one per text, with the token count only as an extra attribute."""
    monkeypatch.setenv("MOCK_EMBEDDING", "true")

    result = await _litellm_engine().embed_text(["a", "b"])

    assert isinstance(result, list)
    assert result == [[0.0, 0.0], [0.0, 0.0]]
    assert result.tokens == 0
