import os
import time
import asyncio
from functools import lru_cache
import logging

from cognee.infrastructure.llm.config import LLMConfig, get_llm_config
from cognee.infrastructure.llm.rate_limiter import EmbeddingRateLimiter
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_embedding_rate_limiting():
    """
    Test the embedding rate limiting feature with a limit of 5 requests per 5 seconds.
    This test sets up the rate limiter, makes a burst of requests, and verifies that
    some are rate limited.
    """
    # Set up environment variables for rate limiting
    os.environ["EMBEDDING_RATE_LIMIT_ENABLED"] = "true"
    os.environ["EMBEDDING_RATE_LIMIT_REQUESTS"] = "5"
    os.environ["EMBEDDING_RATE_LIMIT_INTERVAL"] = "5"
    os.environ["MOCK_EMBEDDING"] = "true"  # Use mock embeddings for testing

    # Clear the config and rate limiter caches to ensure our settings are applied
    get_llm_config.cache_clear()
    EmbeddingRateLimiter.reset_instance()

    # Create a fresh config instance and verify settings
    config = get_llm_config()
    logger.info(f"Embedding Rate Limiting Enabled: {config.embedding_rate_limit_enabled}")
    logger.info(
        f"Embedding Rate Limit: {config.embedding_rate_limit_requests} requests per {config.embedding_rate_limit_interval} seconds"
    )

    # Create an embedding engine
    engine = LiteLLMEmbeddingEngine()

    # Make 10 embedding requests (should be rate limited after 5)
    texts = [f"Test text {i}" for i in range(10)]
    results = []
    failed = 0

    start_time = time.time()

    for i, text in enumerate(texts):
        try:
            logger.info(f"Making request #{i + 1}")
            embedding = await engine.embed_text([text])
            results.append(embedding)
            logger.info(f"Request #{i + 1} succeeded")
        except Exception as e:
            logger.info(f"Request #{i + 1} failed: {e}")
            failed += 1

    end_time = time.time()
    elapsed = end_time - start_time

    # Log results
    logger.info(f"Test completed in {elapsed:.2f} seconds")
    logger.info(f"Made {len(texts)} requests:")
    logger.info(f" - {len(results)} succeeded")
    logger.info(f" - {failed} rate limited")

    # Since we set the limit to 5 requests per 5 seconds, we expect
    # around 5 successful requests and 5 failures
    assert len(results) > 0, "Expected some successful requests"
    assert failed > 0, "Expected some rate limited requests"

    # Reset environment variables
    os.environ.pop("EMBEDDING_RATE_LIMIT_ENABLED", None)
    os.environ.pop("EMBEDDING_RATE_LIMIT_REQUESTS", None)
    os.environ.pop("EMBEDDING_RATE_LIMIT_INTERVAL", None)
    os.environ.pop("MOCK_EMBEDDING", None)


if __name__ == "__main__":
    asyncio.run(test_embedding_rate_limiting())
