from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)


@pytest.mark.asyncio
async def test_embed_text_filters_invalid_inputs(monkeypatch):
    # Use monkeypatch so env is restored automatically
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    engine = LiteLLMEmbeddingEngine(dimensions=4)

    # Stub litellm call to return distinct vectors for each valid input
    UNIQUE_A = [1.0, 1.0, 1.0, 1.0]
    UNIQUE_B = [2.0, 2.0, 2.0, 2.0]
    UNIQUE_C = [2.0, 2.0, 2.0, 2.0]
    UNIQUE_D = [2.0, 2.0, 2.0, 2.0]
    UNIQUE_E = [2.0, 2.0, 2.0, 2.0]

    class _Resp:
        def __init__(self, data):
            self.data = data

    async def fake_aembedding(**kwargs):
        # kwargs["input"] should already be filtered by engine
        # In our inputs below, valid entries are "valid" and "ok!"
        return _Resp(
            [
                {"embedding": UNIQUE_A},
                {"embedding": UNIQUE_B},
                {"embedding": UNIQUE_C},
                {"embedding": UNIQUE_D},
                {"embedding": UNIQUE_E},
            ]
        )

    # Patch the litellm call used in the engine module
    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    inputs = ["", "(", "valid", "   ", "ok!"]
    result = await engine.embed_text(inputs)

    # Output length must match input length
    assert len(result) == len(inputs)

    # Invalid entries should be zero vectors
    assert result[0] == [0.0] * 4
    assert result[3] == [0.0] * 4

    # Valid entries must map to the correct positions
    assert result[1] == UNIQUE_B
    assert result[2] == UNIQUE_C
    assert result[4] == UNIQUE_E


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model", "expected_encoding_format"),
    [
        ("custom", "openrouter/openai/text-embedding-3-small", "float"),
        ("gemini", "gemini/text-embedding-004", None),
    ],
)
async def test_embed_text_sets_encoding_format_only_for_openrouter(
    monkeypatch, provider, model, expected_encoding_format
):
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            provider=provider,
            model=model,
            dimensions=2,
            api_key="test-key",
            endpoint="https://example.invalid/v1",
        )

    response = SimpleNamespace(data=[{"embedding": [0.1, 0.2]}])
    mock_aembedding = AsyncMock(return_value=response)

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", mock_aembedding)

    result = await engine.embed_text(["hello"])

    expected_kwargs = {
        "model": model,
        "input": ["hello"],
        "api_key": "test-key",
        "api_base": "https://example.invalid/v1",
        "api_version": None,
        "dimensions": 2,
    }
    if expected_encoding_format is not None:
        expected_kwargs["encoding_format"] = expected_encoding_format

    mock_aembedding.assert_awaited_once_with(**expected_kwargs)
    assert result == [[0.1, 0.2]]
