import pytest
import asyncio
import os


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment and ensure proper cleanup of async handlers."""
    # Disable Langfuse for tests to avoid async handler warnings
    original_langfuse_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    original_langfuse_secret = os.environ.get("LANGFUSE_SECRET_KEY")

    # Clear Langfuse keys during tests to prevent async handler issues
    if "LANGFUSE_PUBLIC_KEY" in os.environ:
        del os.environ["LANGFUSE_PUBLIC_KEY"]
    if "LANGFUSE_SECRET_KEY" in os.environ:
        del os.environ["LANGFUSE_SECRET_KEY"]

    yield

    # Restore original environment variables after tests
    if original_langfuse_key is not None:
        os.environ["LANGFUSE_PUBLIC_KEY"] = original_langfuse_key
    if original_langfuse_secret is not None:
        os.environ["LANGFUSE_SECRET_KEY"] = original_langfuse_secret


@pytest.fixture(autouse=True)
async def cleanup_async_tasks():
    """Ensure all async tasks are properly cleaned up after each test."""
    yield

    # Wait for any pending async tasks to complete
    try:
        pending = [task for task in asyncio.all_tasks() if not task.done()]
        if pending:
            # Give pending tasks a short time to complete
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=1.0)
    except (asyncio.TimeoutError, RuntimeError):
        # If tasks don't complete within timeout, cancel them
        for task in pending:
            if not task.done():
                task.cancel()

        # Wait for cancelled tasks to finish
        try:
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception:
            pass  # Ignore exceptions from cancelled tasks


@pytest.fixture(autouse=True)
def disable_observability():
    """Disable observability features during tests to prevent async handler issues."""
    from cognee.base_config import get_base_config

    config = get_base_config()
    original_monitoring = config.monitoring_tool

    # Temporarily disable monitoring during tests
    config.monitoring_tool = None

    yield

    # Restore original monitoring setting
    config.monitoring_tool = original_monitoring
