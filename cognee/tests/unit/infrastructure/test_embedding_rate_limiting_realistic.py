import os
import time
import asyncio
from functools import lru_cache
import logging

from cognee.infrastructure.llm.config import LLMConfig, get_llm_config
from cognee.infrastructure.llm.embedding_rate_limiter import EmbeddingRateLimiter
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.tests.unit.infrastructure.mock_embedding_engine import MockEmbeddingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_embedding_rate_limiting_realistic():
    """
    Test the embedding rate limiting feature with a realistic scenario:
    - Set limit to 3 requests per 5 seconds
    - Send requests in bursts with waiting periods
    - Track successful and rate-limited requests
    - Verify the rate limiter's behavior
    """
    # Set up environment variables for rate limiting
    os.environ["EMBEDDING_RATE_LIMIT_ENABLED"] = "true"
    os.environ["EMBEDDING_RATE_LIMIT_REQUESTS"] = "3"  # Only 3 requests per interval
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

    # Create a mock embedding engine
    engine = MockEmbeddingEngine()
    # Configure some delay to simulate realistic API calls but not too long
    engine.configure_mock(add_delay=0.1)

    # Track overall statistics
    total_requests = 0
    total_successes = 0
    total_rate_limited = 0

    # Create a list of tasks to simulate concurrent requests
    async def make_request(i):
        nonlocal total_successes, total_rate_limited
        try:
            logger.info(f"Making request #{i + 1}")
            text = f"Concurrent - Text {i}"
            embedding = await engine.embed_text([text])
            logger.info(f"Request #{i + 1} succeeded with embedding size: {len(embedding[0])}")
            return True
        except Exception as e:
            logger.info(f"Request #{i + 1} rate limited: {e}")
            return False

    # Batch 1: Send 10 concurrent requests (expect 3 to succeed, 7 to be rate limited)
    batch_size = 10
    logger.info(f"\n--- Batch 1: Sending {batch_size} concurrent requests ---")

    batch_start = time.time()
    tasks = [make_request(i) for i in range(batch_size)]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    batch_successes = results.count(True)
    batch_rate_limited = results.count(False)

    batch_end = time.time()
    logger.info(f"Batch 1 completed in {batch_end - batch_start:.2f} seconds")
    logger.info(f"Successes: {batch_successes}, Rate limited: {batch_rate_limited}")

    total_requests += batch_size
    total_successes += batch_successes
    total_rate_limited += batch_rate_limited

    # Wait 2 seconds (should recover some capacity but not all)
    wait_time = 2
    logger.info(f"\nWaiting {wait_time} seconds to allow partial capacity recovery...")
    await asyncio.sleep(wait_time)

    # Batch 2: Send 5 more requests (expect some to succeed, some to be rate limited)
    batch_size = 5
    logger.info(f"\n--- Batch 2: Sending {batch_size} concurrent requests ---")

    batch_start = time.time()
    tasks = [make_request(i) for i in range(batch_size)]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    batch_successes = results.count(True)
    batch_rate_limited = results.count(False)

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

    # Batch 3: Send 3 more requests sequentially (all should succeed)
    batch_size = 3
    logger.info(f"\n--- Batch 3: Sending {batch_size} sequential requests ---")

    batch_start = time.time()
    batch_successes = 0
    batch_rate_limited = 0

    for i in range(batch_size):
        try:
            logger.info(f"Making request #{i + 1}")
            text = f"Sequential - Text {i}"
            embedding = await engine.embed_text([text])
            logger.info(f"Request #{i + 1} succeeded with embedding size: {len(embedding[0])}")
            batch_successes += 1
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
    logger.info("\n--- Test Summary ---")
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


async def test_with_mock_failures():
    """
    Test with the mock engine's ability to generate controlled failures.
    """
    # Setup rate limiting (more permissive settings)
    os.environ["EMBEDDING_RATE_LIMIT_ENABLED"] = "true"
    os.environ["EMBEDDING_RATE_LIMIT_REQUESTS"] = "10"
    os.environ["EMBEDDING_RATE_LIMIT_INTERVAL"] = "5"
    os.environ["DISABLE_RETRIES"] = "true"

    # Clear caches
    get_llm_config.cache_clear()
    EmbeddingRateLimiter.reset_instance()

    # Create a mock engine configured to fail every 3rd request
    engine = MockEmbeddingEngine()
    engine.configure_mock(fail_every_n_requests=3, add_delay=0.1)

    logger.info("\n--- Testing controlled failures with mock ---")

    # Send 10 requests, expecting every 3rd to fail
    for i in range(10):
        try:
            logger.info(f"Making request #{i + 1}")
            text = f"Test text {i}"
            embedding = await engine.embed_text([text])

            logger.info(f"Request #{i + 1} succeeded for {str(embedding)}")
        except Exception as e:
            logger.info(f"Request #{i + 1} failed as expected: {e}")

    # Reset environment variables
    os.environ.pop("EMBEDDING_RATE_LIMIT_ENABLED", None)
    os.environ.pop("EMBEDDING_RATE_LIMIT_REQUESTS", None)
    os.environ.pop("EMBEDDING_RATE_LIMIT_INTERVAL", None)
    os.environ.pop("DISABLE_RETRIES", None)


if __name__ == "__main__":
    asyncio.run(test_embedding_rate_limiting_realistic())
    asyncio.run(test_with_mock_failures())
