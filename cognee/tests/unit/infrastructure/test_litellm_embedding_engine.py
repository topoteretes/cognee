import asyncio
from types import SimpleNamespace

import litellm
import pytest

from cognee.infrastructure.databases.exceptions import EmbeddingException
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)


def test_embed_text_raises_when_provider_returns_empty_embeddings(monkeypatch):
    async def fake_aembedding(**kwargs):
        return SimpleNamespace(data=[])

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)
    monkeypatch.setattr(LiteLLMEmbeddingEngine, "get_tokenizer", lambda self: None)

    engine = LiteLLMEmbeddingEngine(
        model="lm_studio/text-embedding-model",
        provider="custom",
        dimensions=768,
        api_key=None,
        endpoint="http://127.0.0.1:1234/v1/embeddings",
    )

    async def run_test():
        with pytest.raises(EmbeddingException, match="returned no vectors"):
            await engine.embed_text(["ping"])

    asyncio.run(run_test())


def test_embed_text_retries_without_dimensions_on_unsupported_params(monkeypatch):
    calls = []

    async def fake_aembedding(**kwargs):
        calls.append(kwargs)
        if "dimensions" in kwargs:
            raise litellm.exceptions.UnsupportedParamsError(
                message="dimensions unsupported",
                model=kwargs.get("model", ""),
                llm_provider="lm_studio",
            )
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]}])

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)
    monkeypatch.setattr(LiteLLMEmbeddingEngine, "get_tokenizer", lambda self: None)

    engine = LiteLLMEmbeddingEngine(
        model="lm_studio/text-embedding-model",
        provider="custom",
        dimensions=768,
        api_key=None,
        endpoint="http://127.0.0.1:1234/v1/embeddings",
    )

    async def run_test():
        output = await engine.embed_text(["ping"])
        assert output == [[0.1, 0.2, 0.3]]

    asyncio.run(run_test())

    assert len(calls) == 2
    assert "dimensions" in calls[0]
    assert "dimensions" not in calls[1]


def test_embed_text_retries_without_dimensions_on_empty_response(monkeypatch):
    calls = []

    async def fake_aembedding(**kwargs):
        calls.append(kwargs)
        if "dimensions" in kwargs:
            return SimpleNamespace(data=[])
        return SimpleNamespace(data=[{"embedding": [0.4, 0.5]}])

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)
    monkeypatch.setattr(LiteLLMEmbeddingEngine, "get_tokenizer", lambda self: None)

    engine = LiteLLMEmbeddingEngine(
        model="lm_studio/text-embedding-model",
        provider="custom",
        dimensions=768,
        api_key=None,
        endpoint="http://127.0.0.1:1234/v1/embeddings",
    )

    async def run_test():
        output = await engine.embed_text(["ping"])
        assert output == [[0.4, 0.5]]

    asyncio.run(run_test())

    assert len(calls) == 2
    assert "dimensions" in calls[0]
    assert "dimensions" not in calls[1]
