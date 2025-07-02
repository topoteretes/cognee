"""Tests for the LLM rate limiter."""

import pytest
import asyncio
import time
from unittest.mock import patch
from cognee.infrastructure.llm.rate_limiter import (
    llm_rate_limiter,
    rate_limit_async,
    rate_limit_sync,
)


@pytest.fixture(autouse=True)
def reset_limiter_singleton():
    """Reset the singleton instance between tests."""
    llm_rate_limiter._instance = None
    yield


def test_rate_limiter_initialization():
    """Test that the rate limiter can be initialized properly."""
    with patch("cognee.infrastructure.llm.rate_limiter.get_llm_config") as mock_config:
        mock_config.return_value.llm_rate_limit_enabled = True
        mock_config.return_value.llm_rate_limit_requests = 10
        mock_config.return_value.llm_rate_limit_interval = 60  # 1 minute

        limiter = llm_rate_limiter()

        assert limiter._enabled is True
        assert limiter._requests == 10
        assert limiter._interval == 60


def test_rate_limiter_disabled():
    """Test that the rate limiter is disabled by default."""
    with patch("cognee.infrastructure.llm.rate_limiter.get_llm_config") as mock_config:
        mock_config.return_value.llm_rate_limit_enabled = False

        limiter = llm_rate_limiter()

        assert limiter._enabled is False
        assert limiter.hit_limit() is True  # Should always return True when disabled


def test_rate_limiter_singleton():
    """Test that the rate limiter is a singleton."""
    with patch("cognee.infrastructure.llm.rate_limiter.get_llm_config") as mock_config:
        mock_config.return_value.llm_rate_limit_enabled = True
        mock_config.return_value.llm_rate_limit_requests = 5
        mock_config.return_value.llm_rate_limit_interval = 60

        limiter1 = llm_rate_limiter()
        limiter2 = llm_rate_limiter()

        assert limiter1 is limiter2


def test_sync_decorator():
    """Test the sync decorator."""
    with patch("cognee.infrastructure.llm.rate_limiter.llm_rate_limiter") as mock_limiter_class:
        mock_limiter = mock_limiter_class.return_value
        mock_limiter.wait_if_needed.return_value = 0

        @rate_limit_sync
        def test_func():
            return "success"

        result = test_func()

        assert result == "success"
        mock_limiter.wait_if_needed.assert_called_once()


@pytest.mark.asyncio
async def test_async_decorator():
    """Test the async decorator."""
    with patch("cognee.infrastructure.llm.rate_limiter.llm_rate_limiter") as mock_limiter_class:
        mock_limiter = mock_limiter_class.return_value

        # Mock an async method with a coroutine
        async def mock_wait():
            return 0

        mock_limiter.async_wait_if_needed.return_value = mock_wait()

        @rate_limit_async
        async def test_func():
            return "success"

        result = await test_func()

        assert result == "success"
        mock_limiter.async_wait_if_needed.assert_called_once()


def test_rate_limiting_actual():
    """Test actual rate limiting behavior with a small window."""
    with patch("cognee.infrastructure.llm.rate_limiter.get_llm_config") as mock_config:
        # Configure for 3 requests per minute
        mock_config.return_value.llm_rate_limit_enabled = True
        mock_config.return_value.llm_rate_limit_requests = 3
        mock_config.return_value.llm_rate_limit_interval = 60

        # Create a fresh instance
        llm_rate_limiter._instance = None
        limiter = llm_rate_limiter()

        # First 3 requests should succeed
        assert limiter.hit_limit() is True
        assert limiter.hit_limit() is True
        assert limiter.hit_limit() is True

        # Fourth request should fail (exceed limit)
        assert limiter.hit_limit() is False


def test_rate_limit_60_per_minute():
    """Test rate limiting with the default 60 requests per minute limit."""
    with patch("cognee.infrastructure.llm.rate_limiter.get_llm_config") as mock_config:
        # Configure for default values: 60 requests per 60 seconds
        mock_config.return_value.llm_rate_limit_enabled = True
        mock_config.return_value.llm_rate_limit_requests = 60  # 60 requests
        mock_config.return_value.llm_rate_limit_interval = 60  # per minute

        # Create a fresh instance
        llm_rate_limiter._instance = None
        limiter = llm_rate_limiter()

        # Track successful and failed requests
        successes = []
        failures = []

        # Send requests in batches until we see some failures
        # This simulates reaching the rate limit
        num_test_requests = 70  # Try more than our limit of 60

        for i in range(num_test_requests):
            if limiter.hit_limit():
                successes.append(f"Request {i}")
            else:
                failures.append(f"Request {i}")

        # Print the results
        print(f"Total successful requests: {len(successes)}")
        print(f"Total failed requests: {len(failures)}")

        if len(failures) > 0:
            print(f"First failed request: {failures[0]}")

        # Verify we got the expected behavior (close to 60 requests allowed)
        # Allow small variations due to timing
        assert 58 <= len(successes) <= 62, f"Expected ~60 successful requests, got {len(successes)}"
        assert len(failures) > 0, "Expected at least some rate-limited requests"

        # Verify that roughly the first 60 requests succeeded
        if len(failures) > 0:
            first_failure_idx = int(failures[0].split()[1])
            print(f"First failure occurred at request index: {first_failure_idx}")
            assert 58 <= first_failure_idx <= 62, (
                f"Expected first failure around request #60, got #{first_failure_idx}"
            )

        # Calculate requests per minute
        rate_per_minute = len(successes)
        print(f"Rate: {rate_per_minute} requests per minute")

        # Verify the rate is close to our target of 60 requests per minute
        assert 58 <= rate_per_minute <= 62, (
            f"Expected rate of ~60 requests per minute, got {rate_per_minute}"
        )
