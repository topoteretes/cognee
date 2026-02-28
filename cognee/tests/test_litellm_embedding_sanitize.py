import os
import pytest
import asyncio

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)


@pytest.mark.asyncio
async def test_embed_text_filters_invalid_inputs():
    # Enable mock mode so we don't call real APIs
    os.environ["MOCK_EMBEDDING"] = "true"

    engine = LiteLLMEmbeddingEngine(dimensions=4)

    inputs = ["", "(", "valid", "   ", "ok!"]
    result = await engine.embed_text(inputs)

    # Output length must match input length
    assert len(result) == len(inputs)

    # Invalid entries should be zero vectors
    assert result[0] == [0.0] * 4
    assert result[1] == [0.0] * 4
    assert result[3] == [0.0] * 4

    # Valid entries should also be vectors of correct dimension
    assert len(result[2]) == 4
    assert len(result[4]) == 4