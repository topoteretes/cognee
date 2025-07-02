import time
import asyncio
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.llm.rate_limiter import (
    sleep_and_retry_sync,
    sleep_and_retry_async,
    is_rate_limit_error,
)

logger = get_logger()


# Test function to be decorated
@sleep_and_retry_sync(max_retries=3, initial_backoff=0.1, backoff_factor=2.0)
def test_function_sync():
    """A test function that raises rate limit errors a few times, then succeeds."""
    if hasattr(test_function_sync, "counter"):
        test_function_sync.counter += 1
    else:
        test_function_sync.counter = 1

    if test_function_sync.counter <= 2:
        error_msg = "429 Too Many Requests: Rate limit exceeded"
        logger.info(f"Attempt {test_function_sync.counter}: Raising rate limit error")
        raise Exception(error_msg)

    logger.info(f"Attempt {test_function_sync.counter}: Success!")
    return f"Success on attempt {test_function_sync.counter}"


# Test async function to be decorated
@sleep_and_retry_async(max_retries=3, initial_backoff=0.1, backoff_factor=2.0)
async def test_function_async():
    """An async test function that raises rate limit errors a few times, then succeeds."""
    if hasattr(test_function_async, "counter"):
        test_function_async.counter += 1
    else:
        test_function_async.counter = 1

    if test_function_async.counter <= 2:
        error_msg = "429 Too Many Requests: Rate limit exceeded"
        logger.info(f"Attempt {test_function_async.counter}: Raising rate limit error")
        raise Exception(error_msg)

    logger.info(f"Attempt {test_function_async.counter}: Success!")
    return f"Success on attempt {test_function_async.counter}"


def test_is_rate_limit_error():
    """Test the rate limit error detection function."""
    print("\n=== Testing Rate Limit Error Detection ===")

    # Test various error messages that should be detected as rate limit errors
    rate_limit_errors = [
        "429 Rate limit exceeded",
        "Too many requests",
        "rate_limit_exceeded",
        "ratelimit error",
        "You have exceeded your quota",
        "capacity has been exceeded",
        "Service throttled",
    ]

    # Test error messages that should not be detected as rate limit errors
    non_rate_limit_errors = [
        "404 Not Found",
        "500 Internal Server Error",
        "Invalid API Key",
        "Bad Request",
    ]

    # Check that rate limit errors are correctly identified
    for error in rate_limit_errors:
        error_obj = Exception(error)
        result = is_rate_limit_error(error_obj)
        print(f"Error '{error}': {'✓' if result else '✗'} {result}")
        assert result, f"Failed to identify rate limit error: {error}"
        print(f"✓ Correctly identified as rate limit error: {error}")

    # Check that non-rate limit errors are not misidentified
    for error in non_rate_limit_errors:
        error_obj = Exception(error)
        result = is_rate_limit_error(error_obj)
        print(f"Error '{error}': {'✓' if not result else '✗'} {not result}")
        assert not result, f"Incorrectly identified as rate limit error: {error}"
        print(f"✓ Correctly identified as non-rate limit error: {error}")

    print("✅ PASS: Rate limit error detection is working correctly")


def test_sync_retry():
    """Test the synchronous retry decorator."""
    print("\n=== Testing Synchronous Sleep and Retry ===")

    # Reset counter for the test function
    if hasattr(test_function_sync, "counter"):
        del test_function_sync.counter

    # Time the execution to verify backoff is working
    start_time = time.time()

    try:
        result = test_function_sync()
        end_time = time.time()
        elapsed = end_time - start_time

        # Verify results
        print(f"Result: {result}")
        print(f"Test completed in {elapsed:.2f} seconds")
        print(f"Number of attempts: {test_function_sync.counter}")

        # The function should succeed on the 3rd attempt (after 2 failures)
        assert test_function_sync.counter == 3, (
            f"Expected 3 attempts, got {test_function_sync.counter}"
        )
        assert elapsed >= 0.3, f"Expected at least 0.3 seconds of backoff, got {elapsed:.2f}"

        print("✅ PASS: Synchronous retry mechanism is working correctly")
    except Exception as e:
        print(f"❌ FAIL: Test encountered an unexpected error: {str(e)}")
        raise


async def test_async_retry():
    """Test the asynchronous retry decorator."""
    print("\n=== Testing Asynchronous Sleep and Retry ===")

    # Reset counter for the test function
    if hasattr(test_function_async, "counter"):
        del test_function_async.counter

    # Time the execution to verify backoff is working
    start_time = time.time()

    try:
        result = await test_function_async()
        end_time = time.time()
        elapsed = end_time - start_time

        # Verify results
        print(f"Result: {result}")
        print(f"Test completed in {elapsed:.2f} seconds")
        print(f"Number of attempts: {test_function_async.counter}")

        # The function should succeed on the 3rd attempt (after 2 failures)
        assert test_function_async.counter == 3, (
            f"Expected 3 attempts, got {test_function_async.counter}"
        )
        assert elapsed >= 0.3, f"Expected at least 0.3 seconds of backoff, got {elapsed:.2f}"

        print("✅ PASS: Asynchronous retry mechanism is working correctly")
    except Exception as e:
        print(f"❌ FAIL: Test encountered an unexpected error: {str(e)}")
        raise


async def test_retry_max_exceeded():
    """Test what happens when max retries is exceeded."""
    print("\n=== Testing Max Retries Exceeded ===")

    @sleep_and_retry_async(max_retries=2, initial_backoff=0.1)
    async def always_fails():
        """A function that always raises a rate limit error."""
        error_msg = "429 Too Many Requests: Rate limit always exceeded"
        logger.info(f"Always fails with: {error_msg}")
        raise Exception(error_msg)

    try:
        # This should fail after 2 retries (3 attempts total)
        await always_fails()
        print("❌ FAIL: Function should have failed but succeeded")
    except Exception as e:
        print(f"Expected error after max retries: {str(e)}")
        print("✅ PASS: Function correctly failed after max retries exceeded")


async def main():
    """Run all the retry tests."""
    test_is_rate_limit_error()
    test_sync_retry()
    await test_async_retry()
    await test_retry_max_exceeded()

    print("\n=== All Rate Limiting Retry Tests Complete ===")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
