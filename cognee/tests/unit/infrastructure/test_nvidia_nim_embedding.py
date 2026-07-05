from unittest.mock import AsyncMock, Mock, patch

import pytest


class _Resp:
    def __init__(self, data):
        self.data = data


@pytest.mark.asyncio
async def test_nvidia_nim_provider_adds_input_type_and_omits_dimensions(monkeypatch):
    """Provider explicitly set to nvidia_nim: dimensions should be dropped and
    input_type should be forwarded to litellm.aembedding."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
        LiteLLMEmbeddingEngine,
    )

    engine = LiteLLMEmbeddingEngine(
        model="nvidia_nim/nv-embedqa-e5-v5",
        provider="nvidia_nim",
        dimensions=1024,
        input_type="passage",
    )

    captured_kwargs = {}

    async def fake_aembedding(**kwargs):
        captured_kwargs.update(kwargs)
        return _Resp([{"embedding": [1.0, 2.0]}])

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    await engine.embed_text(["hello world"])

    assert "dimensions" not in captured_kwargs
    assert captured_kwargs["input_type"] == "passage"


@pytest.mark.asyncio
async def test_nvidia_nim_provider_inferred_from_model_prefix(monkeypatch):
    """Even if `provider` isn't explicitly "nvidia_nim", a "nvidia_nim/<model>"
    style model string should be enough to trigger the dimensions omission,
    mirroring how litellm itself infers the provider."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch(
        "cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine."
        "LiteLLMEmbeddingEngine.get_tokenizer",
        return_value=Mock(),
    ):
        from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
            LiteLLMEmbeddingEngine,
        )

        engine = LiteLLMEmbeddingEngine(
            model="nvidia_nim/nv-embedqa-e5-v5",
            provider="openai",  # left at default; model prefix should still win
            dimensions=1024,
        )

    captured_kwargs = {}

    async def fake_aembedding(**kwargs):
        captured_kwargs.update(kwargs)
        return _Resp([{"embedding": [1.0, 2.0]}])

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    await engine.embed_text(["hello world"])

    assert "dimensions" not in captured_kwargs


@pytest.mark.asyncio
async def test_non_nvidia_provider_keeps_dimensions_and_skips_input_type(monkeypatch):
    """Sanity check: default OpenAI-style provider is unaffected."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
        LiteLLMEmbeddingEngine,
    )

    engine = LiteLLMEmbeddingEngine(
        model="openai/text-embedding-3-large",
        provider="openai",
        dimensions=3072,
    )

    captured_kwargs = {}

    async def fake_aembedding(**kwargs):
        captured_kwargs.update(kwargs)
        return _Resp([{"embedding": [1.0, 2.0]}])

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    await engine.embed_text(["hello world"])

    assert captured_kwargs["dimensions"] == 3072
    assert "input_type" not in captured_kwargs


@pytest.mark.asyncio
async def test_openai_compatible_engine_forwards_input_type_via_extra_body(monkeypatch):
    """OpenAICompatibleEmbeddingEngine (used for self-hosted NIM-style servers)
    should pass input_type through extra_body without breaking servers that
    don't recognize it."""
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine."
        "OpenAICompatibleEmbeddingEngine.get_tokenizer",
        return_value=Mock(),
    ):
        from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
            OpenAICompatibleEmbeddingEngine,
        )

        engine = OpenAICompatibleEmbeddingEngine(
            model="nv-embedqa-e5-v5",
            dimensions=1024,
            endpoint="http://localhost:8000/v1",
            input_type="query",
        )

    fake_item = Mock(embedding=[1.0, 2.0])
    fake_response = Mock(data=[fake_item])
    engine._client.embeddings.create = AsyncMock(return_value=fake_response)

    await engine.embed_text(["find me a match"])

    _, kwargs = engine._client.embeddings.create.call_args
    assert kwargs["extra_body"] == {"input_type": "query"}
