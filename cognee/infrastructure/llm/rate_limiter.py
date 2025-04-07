"""
Rate limiter for LLM API calls.

This module provides rate limiting functionality for LLM API calls to prevent exceeding
API provider rate limits. The implementation uses the `limits` library with a moving window
strategy to limit requests.

Configuration is done through the LLMConfig with these settings:
- llm_rate_limit_enabled: Whether rate limiting is enabled (default: False)
- llm_rate_limit_requests: Maximum number of requests allowed per interval (default: 60)
- llm_rate_limit_interval: Interval in seconds for the rate limiting window (default: 60)

Usage:
1. Add the decorator to any function that makes API calls:
   @rate_limit_sync
   def my_function():
       # Function that makes API calls

2. For async functions, use the async decorator:
   @rate_limit_async
   async def my_async_function():
       # Async function that makes API calls

3. For automatic retrying on rate limit errors:
   @sleep_and_retry_sync
   def my_function():
       # Function that may experience rate limit errors

4. For async functions with automatic retrying:
   @sleep_and_retry_async
   async def my_async_function():
       # Async function that may experience rate limit errors

5. For embedding rate limiting (uses the same configuration but separate limiter):
   @embedding_rate_limit_async
   async def my_embedding_function():
       # Async function for embedding API calls

6. For embedding with auto-retry:
   @embedding_sleep_and_retry_async
   async def my_embedding_function():
       # Async function for embedding with auto-retry
"""

import time
import asyncio
import random
from functools import wraps
from limits import RateLimitItemPerMinute, storage
from limits.strategies import MovingWindowRateLimiter
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.llm.config import get_llm_config
import threading
import logging
import functools
import openai
import os

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
]

# Default retry settings
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0  # exponential backoff multiplier
DEFAULT_JITTER = 0.1  # 10% jitter to avoid thundering herd


class LLMRateLimiter:
    """
    Rate limiter for LLM API calls.

    This class implements a singleton pattern to ensure that rate limiting
    is consistent across all parts of the application. It uses the limits
    library with a moving window strategy to control request rates.

    The rate limiter converts the configured requests/interval to a per-minute
    rate for compatibility with the limits library's built-in rate limit items.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMRateLimiter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config = get_llm_config()
        self._enabled = self._config.llm_rate_limit_enabled
        self._requests = self._config.llm_rate_limit_requests
        self._interval = self._config.llm_rate_limit_interval

        # Using in-memory storage by default
        self._storage = storage.MemoryStorage()
        self._limiter = MovingWindowRateLimiter(self._storage)

        # Use the built-in per-minute rate limit item
        # We need to adjust the number of requests if interval isn't exactly 60s
        if self._interval == 60:
            self._rate_per_minute = self._requests
        else:
            self._rate_per_minute = int(self._requests * (60 / self._interval))

        self._initialized = True

        if self._enabled:
            logger.info(
                f"LLM Rate Limiter initialized: {self._requests} requests per {self._interval}s"
            )

    def hit_limit(self) -> bool:
        """
        Record a hit and check if limit is exceeded.

        This method checks whether making a request now would exceed the
        configured rate limit. If rate limiting is disabled, it always
        returns True.

        Returns:
            bool: True if the request is allowed, False otherwise.
        """
        if not self._enabled:
            return True

        # Create a fresh rate limit item for each check
        rate_limit = RateLimitItemPerMinute(self._rate_per_minute)

        # Use a consistent key for the API to ensure proper rate limiting
        return self._limiter.hit(rate_limit, "llm_api")

    def wait_if_needed(self) -> float:
        """
        Wait if rate limit is reached.

        This method blocks until the request can be made without exceeding
        the rate limit. It polls every 0.5 seconds.

        Returns:
            float: Time waited in seconds.
        """
        if not self._enabled:
            return 0

        waited = 0
        while not self.hit_limit():
            time.sleep(0.5)
            waited += 0.5

        return waited

    async def async_wait_if_needed(self) -> float:
        """
        Async wait if rate limit is reached.

        This method asynchronously waits until the request can be made without
        exceeding the rate limit. It polls every 0.5 seconds.

        Returns:
            float: Time waited in seconds.
        """
        if not self._enabled:
            return 0

        waited = 0
        while not self.hit_limit():
            await asyncio.sleep(0.5)
            waited += 0.5

        return waited


def rate_limit_sync(func):
    """
    Decorator for rate limiting synchronous functions.

    This decorator ensures that the decorated function respects the
    configured rate limits. If the rate limit would be exceeded,
    the decorator blocks until the request can be made.

    Args:
        func: The synchronous function to decorate.

    Returns:
        The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        limiter = LLMRateLimiter()
        waited = limiter.wait_if_needed()
        if waited > 0:
            logger.debug(f"Rate limited LLM API call, waited for {waited}s")
        return func(*args, **kwargs)

    return wrapper


def rate_limit_async(func):
    """
    Decorator for rate limiting asynchronous functions.

    This decorator ensures that the decorated async function respects the
    configured rate limits. If the rate limit would be exceeded,
    the decorator asynchronously waits until the request can be made.

    Args:
        func: The asynchronous function to decorate.

    Returns:
        The decorated async function.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        limiter = LLMRateLimiter()
        waited = await limiter.async_wait_if_needed()
        if waited > 0:
            logger.debug(f"Rate limited LLM API call, waited for {waited}s")
        return await func(*args, **kwargs)

    return wrapper


def is_rate_limit_error(error):
    """
    Check if an error is related to rate limiting.

    Args:
        error: The exception to check.

    Returns:
        bool: True if the error is rate-limit related, False otherwise.
    """
    error_str = str(error).lower()
    return any(pattern.lower() in error_str for pattern in RATE_LIMIT_ERROR_PATTERNS)


def calculate_backoff(
    attempt,
    initial_backoff=DEFAULT_INITIAL_BACKOFF,
    backoff_factor=DEFAULT_BACKOFF_FACTOR,
    jitter=DEFAULT_JITTER,
):
    """
    Calculate the backoff time for a retry attempt with jitter.

    Args:
        attempt: The current retry attempt (0-based).
        initial_backoff: The initial backoff time in seconds.
        backoff_factor: The multiplier for exponential backoff.
        jitter: The jitter factor to avoid thundering herd.

    Returns:
        float: The backoff time in seconds.
    """
    backoff = initial_backoff * (backoff_factor**attempt)
    jitter_amount = backoff * jitter
    return backoff + random.uniform(-jitter_amount, jitter_amount)


def sleep_and_retry_sync(
    max_retries=DEFAULT_MAX_RETRIES,
    initial_backoff=DEFAULT_INITIAL_BACKOFF,
    backoff_factor=DEFAULT_BACKOFF_FACTOR,
    jitter=DEFAULT_JITTER,
):
    """
    Decorator that automatically retries a synchronous function when rate limit errors occur.

    This decorator implements an exponential backoff strategy with jitter
    to handle rate limit errors efficiently.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_backoff: Initial backoff time in seconds.
        backoff_factor: Multiplier for exponential backoff.
        jitter: Jitter factor to avoid the thundering herd problem.

    Returns:
        The decorated function.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if not is_rate_limit_error(e) or attempt > max_retries:
                        raise

                    backoff_time = calculate_backoff(
                        attempt, initial_backoff, backoff_factor, jitter
                    )
                    logger.warning(
                        f"Rate limit hit, retrying in {backoff_time:.2f}s "
                        f"(attempt {attempt}/{max_retries}): {str(e)}"
                    )
                    time.sleep(backoff_time)

        return wrapper

    return decorator


def sleep_and_retry_async(
    max_retries=DEFAULT_MAX_RETRIES,
    initial_backoff=DEFAULT_INITIAL_BACKOFF,
    backoff_factor=DEFAULT_BACKOFF_FACTOR,
    jitter=DEFAULT_JITTER,
):
    """
    Decorator that automatically retries an asynchronous function when rate limit errors occur.

    This decorator implements an exponential backoff strategy with jitter
    to handle rate limit errors efficiently.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_backoff: Initial backoff time in seconds.
        backoff_factor: Multiplier for exponential backoff.
        jitter: Jitter factor to avoid the thundering herd problem.

    Returns:
        The decorated async function.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if not is_rate_limit_error(e) or attempt > max_retries:
                        raise

                    backoff_time = calculate_backoff(
                        attempt, initial_backoff, backoff_factor, jitter
                    )
                    logger.warning(
                        f"Rate limit hit, retrying in {backoff_time:.2f}s "
                        f"(attempt {attempt}/{max_retries}): {str(e)}"
                    )
                    await asyncio.sleep(backoff_time)

        return wrapper

    return decorator


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
        self.config = get_llm_config()
        self.enabled = self.config.embedding_rate_limit_enabled
        self.requests_limit = self.config.embedding_rate_limit_requests
        self.interval_seconds = self.config.embedding_rate_limit_interval
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
