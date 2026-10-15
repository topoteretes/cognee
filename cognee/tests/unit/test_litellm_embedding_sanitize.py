import base64
import struct
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
    ("provider", "model", "endpoint", "expected_encoding_format"),
    [
        # Documented OpenRouter config (issue #3660): "openrouter/" model prefix.
        (
            "custom",
            "openrouter/openai/text-embedding-3-small",
            "https://example.invalid/v1",
            "float",
        ),
        # Explicit provider name.
        ("openrouter", "text-embedding-3-small", "https://example.invalid/v1", "float"),
        # Prefix detection is case-insensitive.
        (
            "custom",
            "OpenRouter/openai/text-embedding-3-small",
            "https://example.invalid/v1",
            "float",
        ),
        # Custom endpoint aimed at openrouter.ai with an unprefixed model: litellm
        # drives this through its OpenAI handler (the branch that injected the null),
        # so the guard must still fire based on the endpoint host.
        ("custom", "openai/text-embedding-3-small", "https://openrouter.ai/api/v1", "float"),
        # OpenAI and Azure OpenAI routes ask for base64 vectors.
        ("openai", "openai/text-embedding-3-large", None, "base64"),
        ("openai", "text-embedding-3-large", None, "base64"),
        ("openai", "openai/text-embedding-3-large", "https://proxy.invalid/v1", "base64"),
        ("azure", "azure/my-embedding-deployment", "https://example.invalid", "base64"),
        # Providers that reject encoding_format must be left untouched (omitted).
        ("gemini", "gemini/text-embedding-004", "https://example.invalid/v1", None),
    ],
)
async def test_embed_text_sets_encoding_format_per_route(
    monkeypatch, provider, model, endpoint, expected_encoding_format
):
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            provider=provider,
            model=model,
            dimensions=2,
            api_key="test-key",
            endpoint=endpoint,
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
        "api_base": endpoint,
        "api_version": None,
        "dimensions": 2,
    }
    if expected_encoding_format is not None:
        expected_kwargs["encoding_format"] = expected_encoding_format

    mock_aembedding.assert_awaited_once_with(**expected_kwargs)
    assert result == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_embed_text_decodes_base64_vectors(monkeypatch):
    """A base64 payload decodes to the float32 values it carries; float lists pass through."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            provider="openai",
            model="openai/text-embedding-3-large",
            dimensions=4,
            api_key="test-key",
        )

    vector = [0.0123, -0.5, 0.25, 1e-7]
    float32_vector = [struct.unpack("<f", struct.pack("<f", v))[0] for v in vector]
    encoded = base64.b64encode(struct.pack("<4f", *vector)).decode()
    response = SimpleNamespace(data=[{"embedding": encoded}, {"embedding": float32_vector}])

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", AsyncMock(return_value=response))

    result = await engine.embed_text(["a", "b"])

    assert result == [float32_vector, float32_vector]
    assert all(isinstance(v, float) for v in result[0])
