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


class llm_rate_limiter:
    """
    Rate limiter for LLM API calls.

    This class implements a singleton pattern to ensure that rate limiting
    is consistent across all parts of the application. It uses the limits
    library with a moving window strategy to control request rates.

    The rate limiter converts the configured requests/interval to a per-minute
    rate for compatibility with the limits library's built-in rate limit items.

    Public methods:
    - hit_limit
    - wait_if_needed
    - async_wait_if_needed

    Instance variables:
    - _enabled
    - _requests
    - _interval
    - _storage
    - _limiter
    - _rate_per_minute
    - _initialized
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(llm_rate_limiter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        config = get_llm_config()
        self._enabled = config.llm_rate_limit_enabled
        self._requests = config.llm_rate_limit_requests
        self._interval = config.llm_rate_limit_interval

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

        Returns:
        --------

            - bool: True if the request is allowed, False otherwise.
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

        Returns:
        --------

            - float: Time waited in seconds.
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

        Returns:
        --------

            - float: Time waited in seconds.
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

    Parameters:
    -----------

        - func: The synchronous function to decorate.

    Returns:
    --------

        The decorated function that applies rate limiting to the original function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Manage rate limiting for the wrapped function's execution.

        This decorator checks if the rate limit for LLM API calls is exceeded before executing
        the function. It waits if necessary and logs the time waited. The rate limiter instance
        is obtained from the llm_rate_limiter class, ensuring consistent access to rate limiting
        behavior across function calls.

        Parameters:
        -----------

            - *args: Positional arguments passed to the wrapped function.
            - **kwargs: Keyword arguments passed to the wrapped function.

        Returns:
        --------

            The return value of the wrapped function after applying the rate limiting logic.
        """
        limiter = llm_rate_limiter()
        waited = limiter.wait_if_needed()
        if waited > 0:
            logger.debug(f"Rate limited LLM API call, waited for {waited}s")
        return func(*args, **kwargs)

    return wrapper


def rate_limit_async(func):
    """
    Decorate an asynchronous function for rate limiting.

    This decorator ensures that the decorated async function respects the configured rate
    limits. If the rate limit would be exceeded, the decorator asynchronously waits until
    the request can be made.

    Parameters:
    -----------

        - func: The asynchronous function to decorate.

    Returns:
    --------

        The decorated async function.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        """
        Handle rate limiting for LLM API calls in an asynchronous context.

        This wrapper function first ensures that the LLM rate limiter is utilized before
        executing the provided function. It waits if necessary to adhere to any configured rate
        limits and logs the waiting duration if applicable.

        Parameters:
        -----------

            - *args: Positional arguments passed to the wrapped function.
            - **kwargs: Keyword arguments passed to the wrapped function.

        Returns:
        --------

            The return value of the wrapped function, after ensuring rate limiting compliance.
        """
        limiter = llm_rate_limiter()
        waited = await limiter.async_wait_if_needed()
        if waited > 0:
            logger.debug(f"Rate limited LLM API call, waited for {waited}s")
        return await func(*args, **kwargs)

    return wrapper


def is_rate_limit_error(error):
    """
    Check if an error is related to rate limiting.

    Evaluate the provided error to determine if it signifies a rate limiting issue by
    checking against known patterns. The check is case-insensitive and looks for matches in
    the string representation of the error.

    Parameters:
    -----------

        - error: The exception to check for rate limiting indications.

    Returns:
    --------

        - bool: True if the error is rate-limit related, False otherwise.
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

    Parameters:
    -----------

        - attempt: The current retry attempt (0-based).
        - initial_backoff: The initial backoff time in seconds. (default
          DEFAULT_INITIAL_BACKOFF)
        - backoff_factor: The multiplier for exponential backoff. (default
          DEFAULT_BACKOFF_FACTOR)
        - jitter: The jitter factor to avoid thundering herd. (default DEFAULT_JITTER)

    Returns:
    --------

        The backoff time in seconds, calculated using the exponential backoff formula
        adjusted by a jitter component.
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
    Decorate a synchronous function to automatically retry on rate limit errors.

    This decorator implements an exponential backoff strategy with jitter to handle rate
    limit errors efficiently. It will retry the decorated function until success or until
    the maximum number of retries is reached.

    Parameters:
    -----------

        - max_retries: Maximum number of retry attempts. (default DEFAULT_MAX_RETRIES)
        - initial_backoff: Initial backoff time in seconds. (default
          DEFAULT_INITIAL_BACKOFF)
        - backoff_factor: Multiplier for exponential backoff. (default
          DEFAULT_BACKOFF_FACTOR)
        - jitter: Jitter factor to avoid the thundering herd problem. (default
          DEFAULT_JITTER)

    Returns:
    --------

        The decorated function that retries on rate limit errors.
    """

    def decorator(func):
        """
        Apply a retry mechanism to a function, specifically for handling rate limit errors.

        Parameters:
        -----------

            - func: The function to be wrapped with retry logic.

        Returns:
        --------

            A wrapped function that retries on rate limit errors until successful or max retries
            are reached.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Handle retries for a function call in case of errors, specifically for rate limit
            errors.

            This decorator will continually attempt to call the provided function until it either
            succeeds or the maximum number of retries is reached. If the caught exception is
            determined to be a rate limit error, it will calculate the appropriate backoff time and
            wait before retrying.

            Parameters:
            -----------

                - *args: Positional arguments to be passed to the wrapped function.
                - **kwargs: Keyword arguments to be passed to the wrapped function.

            Returns:
            --------

                The return value of the wrapped function if the call succeeds before exceeding
                max_retries.
            """
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
    Retry an asynchronous function with an exponential backoff strategy upon encountering
    rate limit errors.

    This decorator will automatically retry the specified asynchronous function if it raises
    an error related to rate limiting. It uses a backoff strategy to control the timing of
    retries, which includes a random jitter to prevent overwhelming the server.

    Args:
    max_retries: Maximum number of retry attempts.
    initial_backoff: Initial backoff time in seconds.
    backoff_factor: Multiplier for exponential backoff.
    jitter: Jitter factor to avoid the thundering herd problem.

    Returns:
    The decorated async function that handles retries on rate limit errors, returning the
    function's original output after successful completion.
    """

    def decorator(func):
        """
        Wrap an asynchronous function to handle retries with a backoff strategy.

        Parameters:
        -----------

            - func: The asynchronous function to be wrapped and retried.

        Returns:
        --------

            The wrapped asynchronous function that handles retries.
        """

        @wraps(func)
        async def wrapper(*args, **kwargs):
            """
            Handle retries for a given async function in case of errors, with rate limit check and
            backoff strategy.

            Wrap the provided asynchronous function and execute it with retries based on error
            handling. If a rate limit error occurs, retry the function after calculating an
            appropriate backoff time, respecting the maximum number of retries allowed. In case of
            non-rate limit errors or exceeding max retries, re-raise the exception.

            Parameters:
            -----------

                - *args: Positional arguments to be passed to the wrapped async function.
                - **kwargs: Keyword arguments to be passed to the wrapped async function.

            Returns:
            --------

                The result of the wrapped async function upon successful completion after retries.
            """
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
