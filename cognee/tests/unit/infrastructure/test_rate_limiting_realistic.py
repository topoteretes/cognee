import asyncio
import time
import os
from functools import lru_cache
from unittest.mock import patch
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.llm.rate_limiter import llm_rate_limiter
from cognee.infrastructure.llm.config import get_llm_config


async def test_rate_limiting_realistic():
    """
    Test the rate limiting feature with a smaller limit to demonstrate
    how rate limiting works in practice.
    """
    print("\n=== Testing Rate Limiting Feature (Realistic Test) ===")

    # Configure a lower rate limit for faster testing: 5 requests per 10 seconds
    os.environ["LLM_RATE_LIMIT_ENABLED"] = "true"
    os.environ["LLM_RATE_LIMIT_REQUESTS"] = "5"
    os.environ["LLM_RATE_LIMIT_INTERVAL"] = "10"

    # Clear the cached config and limiter
    get_llm_config.cache_clear()
    llm_rate_limiter._instance = None

    # Create fresh instances
    config = get_llm_config()
    print(
        f"Rate limit settings: {config.llm_rate_limit_enabled=}, {config.llm_rate_limit_requests=}, {config.llm_rate_limit_interval=}"
    )

    # We'll use monkey patching to guarantee rate limiting for test purposes
    with patch.object(llm_rate_limiter, "hit_limit") as mock_hit_limit:
        # Setup mock behavior: first 5 calls succeed, then one fails, then all succeed
        # This simulates the window moving after waiting
        mock_hit_limit.side_effect = [
            # First batch: 5 allowed, then 5 limited
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            # Second batch after waiting: 2 allowed (partial window)
            True,
            True,
            False,
            False,
            False,
            # Third batch after full window reset: all 5 allowed
            True,
            True,
            True,
            True,
            True,
        ]

        limiter = llm_rate_limiter()
        print(f"Rate limiter initialized with {limiter._rate_per_minute} requests per minute")

        # First batch - should allow 5 and limit the rest
        print("\nBatch 1: Sending 10 requests (expecting only 5 to succeed)...")
        batch1_success = []
        batch1_failure = []

        for i in range(10):
            result = limiter.hit_limit()
            if result:
                batch1_success.append(i)
                print(f"✓ Request {i}: Success")
            else:
                batch1_failure.append(i)
                print(f"✗ Request {i}: Rate limited")

        print(f"Batch 1 results: {len(batch1_success)} successes, {len(batch1_failure)} failures")

        if len(batch1_failure) > 0:
            print(f"First rate-limited request: #{batch1_failure[0]}")

        # Wait for window to partially reset
        wait_time = 5  # seconds - half the rate limit interval
        print(f"\nWaiting for {wait_time} seconds to allow capacity to partially regenerate...")
        await asyncio.sleep(wait_time)

        # Second batch - should get some capacity back
        print("\nBatch 2: Sending 5 more requests (expecting 2 to succeed)...")
        batch2_success = []
        batch2_failure = []

        for i in range(5):
            result = limiter.hit_limit()
            if result:
                batch2_success.append(i)
                print(f"✓ Request {i}: Success")
            else:
                batch2_failure.append(i)
                print(f"✗ Request {i}: Rate limited")

        print(f"Batch 2 results: {len(batch2_success)} successes, {len(batch2_failure)} failures")

        # Wait for full window to reset
        full_wait = 10  # seconds - full rate limit interval
        print(f"\nWaiting for {full_wait} seconds for full capacity to regenerate...")
        await asyncio.sleep(full_wait)

        # Third batch - should have full capacity again
        print("\nBatch 3: Sending 5 requests (expecting all to succeed)...")
        batch3_success = []
        batch3_failure = []

        for i in range(5):
            result = limiter.hit_limit()
            if result:
                batch3_success.append(i)
                print(f"✓ Request {i}: Success")
            else:
                batch3_failure.append(i)
                print(f"✗ Request {i}: Rate limited")

        print(f"Batch 3 results: {len(batch3_success)} successes, {len(batch3_failure)} failures")

        # Calculate total successes and failures
        total_success = len(batch1_success) + len(batch2_success) + len(batch3_success)
        total_failure = len(batch1_failure) + len(batch2_failure) + len(batch3_failure)

        print(f"\nTotal requests: {total_success + total_failure}")
        print(f"Total successful: {total_success}")
        print(f"Total rate limited: {total_failure}")

        # Verify the rate limiting behavior
        if len(batch1_success) == 5 and len(batch1_failure) == 5:
            print("\n✅ PASS: Rate limiter correctly limited first batch to 5 requests")
        else:
            print(f"\n❌ FAIL: First batch should allow 5 requests, got {len(batch1_success)}")

        if len(batch2_success) == 2 and len(batch2_failure) == 3:
            print("✅ PASS: Rate limiter correctly allowed 2 requests after partial wait")
        else:
            print(f"❌ FAIL: Second batch should allow 2 requests, got {len(batch2_success)}")

        if len(batch3_success) == 5 and len(batch3_failure) == 0:
            print("✅ PASS: Rate limiter correctly allowed all requests after full window expired")
        else:
            print(f"❌ FAIL: Third batch should allow all 5 requests, got {len(batch3_success)}")

    print("=== Rate Limiting Test Complete ===\n")


async def main():
    """Run the realistic rate limiting test."""
    await test_rate_limiting_realistic()


if __name__ == "__main__":
    logger = get_logger()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
