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


async def test_embedding_rate_limiting_realistic():
    """
    Test the embedding rate limiting feature with a realistic scenario:
    - Set limit to 5 requests per 5 seconds
    - Send requests in bursts with waiting periods
    - Track successful and rate-limited requests
    - Verify the rate limiter's behavior
    """
    # Set up environment variables for rate limiting
    os.environ["EMBEDDING_RATE_LIMIT_ENABLED"] = "true"
    os.environ["EMBEDDING_RATE_LIMIT_REQUESTS"] = "5"
    os.environ["EMBEDDING_RATE_LIMIT_INTERVAL"] = "5"
    os.environ["MOCK_EMBEDDING"] = "true"  # Use mock embeddings for testing
    os.environ["DISABLE_RETRIES"] = "true"  # Disable automatic retries for testing

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

    # Track overall statistics
    total_requests = 0
    total_successes = 0
    total_rate_limited = 0

    # Batch 1: Send 8 requests (expect 5 to succeed, 3 to be rate limited)
    batch_size = 8
    logger.info(f"\n--- Batch 1: Sending {batch_size} requests ---")
    texts = [f"Batch 1 - Text {i}" for i in range(batch_size)]

    batch_start = time.time()
    batch_successes = 0
    batch_rate_limited = 0

    for i, text in enumerate(texts):
        try:
            logger.info(f"Making request #{i + 1}")
            embedding = await engine.embed_text([text])
            batch_successes += 1
            logger.info(f"Request #{i + 1} succeeded")
        except Exception as e:
            logger.info(f"Request #{i + 1} rate limited: {e}")
            batch_rate_limited += 1

    batch_end = time.time()
    logger.info(f"Batch 1 completed in {batch_end - batch_start:.2f} seconds")
    logger.info(f"Successes: {batch_successes}, Rate limited: {batch_rate_limited}")

    total_requests += batch_size
    total_successes += batch_successes
    total_rate_limited += batch_rate_limited

    # Wait 3 seconds (should recover some capacity but not all)
    wait_time = 3
    logger.info(f"\nWaiting {wait_time} seconds to allow partial capacity recovery...")
    await asyncio.sleep(wait_time)

    # Batch 2: Send 5 more requests (expect some to succeed, some to be rate limited)
    batch_size = 5
    logger.info(f"\n--- Batch 2: Sending {batch_size} requests ---")
    texts = [f"Batch 2 - Text {i}" for i in range(batch_size)]

    batch_start = time.time()
    batch_successes = 0
    batch_rate_limited = 0

    for i, text in enumerate(texts):
        try:
            logger.info(f"Making request #{i + 1}")
            embedding = await engine.embed_text([text])
            batch_successes += 1
            logger.info(f"Request #{i + 1} succeeded")
        except Exception as e:
            logger.info(f"Request #{i + 1} rate limited: {e}")
            batch_rate_limited += 1

    batch_end = time.time()
    logger.info(f"Batch 2 completed in {batch_end - batch_start:.2f} seconds")
    logger.info(f"Successes: {batch_successes}, Rate limited: {batch_rate_limited}")

    total_requests += batch_size
    total_successes += batch_successes
    total_rate_limited += batch_rate_limited

    # Wait 5 seconds (should recover full capacity)
    wait_time = 5
    logger.info(f"\nWaiting {wait_time} seconds to allow full capacity recovery...")
    await asyncio.sleep(wait_time)

    # Batch 3: Send 5 more requests (all should succeed)
    batch_size = 5
    logger.info(f"\n--- Batch 3: Sending {batch_size} requests ---")
    texts = [f"Batch 3 - Text {i}" for i in range(batch_size)]

    batch_start = time.time()
    batch_successes = 0
    batch_rate_limited = 0

    for i, text in enumerate(texts):
        try:
            logger.info(f"Making request #{i + 1}")
            embedding = await engine.embed_text([text])
            batch_successes += 1
            logger.info(f"Request #{i + 1} succeeded")
        except Exception as e:
            logger.info(f"Request #{i + 1} rate limited: {e}")
            batch_rate_limited += 1

    batch_end = time.time()
    logger.info(f"Batch 3 completed in {batch_end - batch_start:.2f} seconds")
    logger.info(f"Successes: {batch_successes}, Rate limited: {batch_rate_limited}")

    total_requests += batch_size
    total_successes += batch_successes
    total_rate_limited += batch_rate_limited

    # Log overall results
    logger.info(f"\n--- Test Summary ---")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Total successes: {total_successes}")
    logger.info(f"Total rate limited: {total_rate_limited}")

    # Verify the behavior
    assert total_successes > 0, "Expected some successful requests"
    assert total_rate_limited > 0, "Expected some rate limited requests"

    # Reset environment variables
    os.environ.pop("EMBEDDING_RATE_LIMIT_ENABLED", None)
    os.environ.pop("EMBEDDING_RATE_LIMIT_REQUESTS", None)
    os.environ.pop("EMBEDDING_RATE_LIMIT_INTERVAL", None)
    os.environ.pop("MOCK_EMBEDDING", None)
    os.environ.pop("DISABLE_RETRIES", None)


if __name__ == "__main__":
    asyncio.run(test_embedding_rate_limiting_realistic())
