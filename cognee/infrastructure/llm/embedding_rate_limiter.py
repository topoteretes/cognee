import threading
import logging
import functools
import os
import time
import asyncio
import random
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.llm.config import get_llm_config


logger = get_logger()

# Common error patterns that indicate rate limiting
RATE_LIMIT_ERROR_PATTERNS = [
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "retry after",
    "capacity",
    "quota",
    "limit exceeded",
    "tps limit exceeded",
    "request limit exceeded",
    "maximum requests",
    "exceeded your current quota",
    "throttled",
    "throttling",
]

# Default retry settings
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0  # exponential backoff multiplier
DEFAULT_JITTER = 0.1  # 10% jitter to avoid thundering herd


class EmbeddingRateLimiter:
    """
    Rate limiter for embedding API calls.

    This class implements a singleton pattern to ensure that rate limiting
    is consistent across all embedding requests. It uses the limits
    library with a moving window strategy to control request rates.

    The rate limiter uses the same configuration as the LLM API rate limiter
    but uses a separate key to track embedding API calls independently.
    """

    _instance = None
    lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls.lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        with cls.lock:
            cls._instance = None

    def __init__(self):
        config = get_llm_config()
        self.enabled = config.embedding_rate_limit_enabled
        self.requests_limit = config.embedding_rate_limit_requests
        self.interval_seconds = config.embedding_rate_limit_interval
        self.request_times = []
        self.lock = threading.Lock()

        logging.info(
            f"EmbeddingRateLimiter initialized: enabled={self.enabled}, "
            f"requests_limit={self.requests_limit}, interval_seconds={self.interval_seconds}"
        )

    def hit_limit(self) -> bool:
        """
        Check if the current request would exceed the rate limit.

        Returns:
            bool: True if the rate limit would be exceeded, False otherwise.
        """
        if not self.enabled:
            return False

        current_time = time.time()

        with self.lock:
            # Remove expired request times
            cutoff_time = current_time - self.interval_seconds
            self.request_times = [t for t in self.request_times if t > cutoff_time]

            # Check if adding a new request would exceed the limit
            if len(self.request_times) >= self.requests_limit:
                logger.info(
                    f"Rate limit hit: {len(self.request_times)} requests in the last {self.interval_seconds} seconds"
                )
                return True

            # Otherwise, we're under the limit
            return False

    def wait_if_needed(self) -> float:
        """
        Block until a request can be made without exceeding the rate limit.

        Returns:
            float: Time waited in seconds.
        """
        if not self.enabled:
            return 0

        wait_time = 0
        start_time = time.time()

        while self.hit_limit():
            time.sleep(0.5)  # Poll every 0.5 seconds
            wait_time = time.time() - start_time

        # Record this request
        with self.lock:
            self.request_times.append(time.time())

        return wait_time

    async def async_wait_if_needed(self) -> float:
        """
        Asynchronously wait until a request can be made without exceeding the rate limit.

        Returns:
            float: Time waited in seconds.
        """
        if not self.enabled:
            return 0

        wait_time = 0
        start_time = time.time()

        while self.hit_limit():
            await asyncio.sleep(0.5)  # Poll every 0.5 seconds
            wait_time = time.time() - start_time

        # Record this request
        with self.lock:
            self.request_times.append(time.time())

        return wait_time


def embedding_rate_limit_sync(func):
    """
    Decorator that applies rate limiting to a synchronous embedding function.

    This decorator checks if the request would exceed the rate limit,
    and blocks if necessary.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function that applies rate limiting.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        limiter = EmbeddingRateLimiter.get_instance()

        # Check if rate limiting is enabled and if we're at the limit
        if limiter.hit_limit():
            error_msg = "Embedding API rate limit exceeded"
            logger.warning(error_msg)

            # Create a custom embedding rate limit exception
            from cognee.infrastructure.databases.exceptions.EmbeddingException import (
                EmbeddingException,
            )

            raise EmbeddingException(error_msg)

        # Add this request to the counter and proceed
        limiter.wait_if_needed()
        return func(*args, **kwargs)

    return wrapper


def embedding_rate_limit_async(func):
    """
    Decorator that applies rate limiting to an asynchronous embedding function.

    This decorator checks if the request would exceed the rate limit,
    and waits asynchronously if necessary.

    Args:
        func: Async function to decorate.

    Returns:
        Decorated async function that applies rate limiting.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        limiter = EmbeddingRateLimiter.get_instance()

        # Check if rate limiting is enabled and if we're at the limit
        if limiter.hit_limit():
            error_msg = "Embedding API rate limit exceeded"
            logger.warning(error_msg)

            # Create a custom embedding rate limit exception
            from cognee.infrastructure.databases.exceptions.EmbeddingException import (
                EmbeddingException,
            )

            raise EmbeddingException(error_msg)

        # Add this request to the counter and proceed
        await limiter.async_wait_if_needed()
        return await func(*args, **kwargs)

    return wrapper


def embedding_sleep_and_retry_sync(max_retries=5, base_backoff=1.0, jitter=0.5):
    """
    Decorator that adds retry with exponential backoff for synchronous embedding functions.

    The decorator will retry the function with exponential backoff if it
    fails due to a rate limit error.

    Args:
        max_retries: Maximum number of retries.
        base_backoff: Base backoff time in seconds.
        jitter: Jitter factor to randomize backoff time.

    Returns:
        Decorated function that retries on rate limit errors.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If DISABLE_RETRIES is set, don't retry for testing purposes
            disable_retries = os.environ.get("DISABLE_RETRIES", "false").lower() in (
                "true",
                "1",
                "yes",
            )

            retries = 0
            last_error = None

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if this is a rate limit error
                    error_str = str(e).lower()
                    error_type = type(e).__name__
                    is_rate_limit = any(
                        pattern in error_str.lower() for pattern in RATE_LIMIT_ERROR_PATTERNS
                    )

                    if disable_retries:
                        # For testing, propagate the exception immediately
                        raise

                    if is_rate_limit and retries < max_retries:
                        # Calculate backoff with jitter
                        backoff = (
                            base_backoff * (2**retries) * (1 + random.uniform(-jitter, jitter))
                        )

                        logger.warning(
                            f"Embedding rate limit hit, retrying in {backoff:.2f}s "
                            f"(attempt {retries + 1}/{max_retries}): "
                            f"({error_str!r}, {error_type!r})"
                        )

                        time.sleep(backoff)
                        retries += 1
                        last_error = e
                    else:
                        # Not a rate limit error or max retries reached, raise
                        raise

            # If we exit the loop due to max retries, raise the last error
            if last_error:
                raise last_error

        return wrapper

    return decorator


def embedding_sleep_and_retry_async(max_retries=5, base_backoff=1.0, jitter=0.5):
    """
    Decorator that adds retry with exponential backoff for asynchronous embedding functions.

    The decorator will retry the function with exponential backoff if it
    fails due to a rate limit error.

    Args:
        max_retries: Maximum number of retries.
        base_backoff: Base backoff time in seconds.
        jitter: Jitter factor to randomize backoff time.

    Returns:
        Decorated async function that retries on rate limit errors.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # If DISABLE_RETRIES is set, don't retry for testing purposes
            disable_retries = os.environ.get("DISABLE_RETRIES", "false").lower() in (
                "true",
                "1",
                "yes",
            )

            retries = 0
            last_error = None

            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if this is a rate limit error
                    error_str = str(e).lower()
                    error_type = type(e).__name__
                    is_rate_limit = any(
                        pattern in error_str.lower() for pattern in RATE_LIMIT_ERROR_PATTERNS
                    )

                    if disable_retries:
                        # For testing, propagate the exception immediately
                        raise

                    if is_rate_limit and retries < max_retries:
                        # Calculate backoff with jitter
                        backoff = (
                            base_backoff * (2**retries) * (1 + random.uniform(-jitter, jitter))
                        )

                        logger.warning(
                            f"Embedding rate limit hit, retrying in {backoff:.2f}s "
                            f"(attempt {retries + 1}/{max_retries}): "
                            f"({error_str!r}, {error_type!r})"
                        )

                        await asyncio.sleep(backoff)
                        retries += 1
                        last_error = e
                    else:
                        # Not a rate limit error or max retries reached, raise
                        raise

            # If we exit the loop due to max retries, raise the last error
            if last_error:
                raise last_error

        return wrapper

    return decorator
