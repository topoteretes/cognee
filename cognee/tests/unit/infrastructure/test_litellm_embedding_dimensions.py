import os
from unittest.mock import patch

import pytest

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import LiteLLMEmbeddingEngine

@pytest.mark.asyncio
async def test_litellm_embedding_custom_dimensions():
    """
    Test that LiteLLMEmbeddingEngine correctly respects the 'dimensions' parameter
    in mock mode.
    """
    # Force mock mode for this test
    with patch.dict(os.environ, {"MOCK_EMBEDDING": "true"}):
        custom_dim = 1024
        engine = LiteLLMEmbeddingEngine(dimensions=custom_dim)
        
        text = ["Hello world"]
        embeddings = await engine.embed_text(text)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == custom_dim, f"Expected dimension {custom_dim}, but got {len(embeddings[0])}"

@pytest.mark.asyncio
async def test_litellm_embedding_default_dimensions():
    """
    Test that LiteLLMEmbeddingEngine uses the default dimension (3072) 
    when no dimension is provided.
    """
    with patch.dict(os.environ, {"MOCK_EMBEDDING": "true"}):
        engine = LiteLLMEmbeddingEngine(dimensions=None)
        
        text = ["Hello world"]
        embeddings = await engine.embed_text(text)
        
        expected_default = 3072
        assert len(embeddings) == 1
        assert len(embeddings[0]) == expected_default, f"Expected default dimension {expected_default}, but got {len(embeddings[0])}"

@pytest.mark.asyncio
async def test_litellm_embedding_invalid_dimensions():
    """
    Test that LiteLLMEmbeddingEngine raises ValueError for invalid dimensions.
    """
    with pytest.raises(ValueError, match="dimensions must be a positive integer"):
        LiteLLMEmbeddingEngine(dimensions=0)
        
    with pytest.raises(ValueError, match="dimensions must be a positive integer"):
        LiteLLMEmbeddingEngine(dimensions=-100)
    
    with pytest.raises(ValueError, match="dimensions must be a positive integer"):
        LiteLLMEmbeddingEngine(dimensions="1024") # type: ignore

    with pytest.raises(ValueError, match="dimensions must be a positive integer"):
        LiteLLMEmbeddingEngine(dimensions=1024.5) # type: ignore