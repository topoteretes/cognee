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

    Public Methods:
    - get_instance
    - reset_instance
    - hit_limit
    - wait_if_needed
    - async_wait_if_needed

    Instance Variables:
    - enabled
    - requests_limit
    - interval_seconds
    - request_times
    - lock
    """

    _instance = None
    lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """
        Retrieve the singleton instance of the EmbeddingRateLimiter.

        This method ensures that only one instance of the class exists and
        is thread-safe. It lazily initializes the instance if it doesn't
        already exist.

        Returns:
        --------

            The singleton instance of the EmbeddingRateLimiter class.
        """
        if cls._instance is None:
            with cls.lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance of the EmbeddingRateLimiter.

        This method is thread-safe and sets the instance to None, allowing
        for a new instance to be created when requested again.
        """
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

        This method checks if the rate limiter is enabled and evaluates
        the number of requests made in the elapsed interval.

        Returns:
        - bool: True if the rate limit would be exceeded, False otherwise.

        Returns:
        --------

            - bool: True if the rate limit would be exceeded, otherwise False.
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

        This method will wait if the current request would exceed the
        rate limit and returns the time waited in seconds.

        Returns:
        - float: Time waited in seconds before a request is allowed.

        Returns:
        --------

            - float: Time waited in seconds before proceeding.
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

        This method will wait if the current request would exceed the
        rate limit and returns the time waited in seconds.

        Returns:
        - float: Time waited in seconds before a request is allowed.

        Returns:
        --------

            - float: Time waited in seconds before proceeding.
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
    Apply rate limiting to a synchronous embedding function.

    Parameters:
    -----------

        - func: Function to decorate with rate limiting logic.

    Returns:
    --------

        Returns the decorated function that applies rate limiting.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrap the given function with rate limiting logic to control the embedding API usage.

        Checks if the rate limit has been exceeded before allowing the function to execute. If
        the limit is hit, it logs a warning and raises an EmbeddingException. Otherwise, it
        updates the request count and proceeds to call the original function.

        Parameters:
        -----------

            - *args: Variable length argument list for the wrapped function.
            - **kwargs: Keyword arguments for the wrapped function.

        Returns:
        --------

            Returns the result of the wrapped function if rate limiting conditions are met.
        """
        limiter = EmbeddingRateLimiter.get_instance()

        # Check if rate limiting is enabled and if we're at the limit
        if limiter.hit_limit():
            error_msg = "Embedding API rate limit exceeded"
            logger.warning(error_msg)

            # Create a custom embedding rate limit exception
            from cognee.infrastructure.databases.exceptions import EmbeddingException

            raise EmbeddingException(error_msg)

        # Add this request to the counter and proceed
        limiter.wait_if_needed()
        return func(*args, **kwargs)

    return wrapper


def embedding_rate_limit_async(func):
    """
    Decorator that applies rate limiting to an asynchronous embedding function.

    Parameters:
    -----------

        - func: Async function to decorate.

    Returns:
    --------

        Returns the decorated async function that applies rate limiting.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        """
        Handle function calls with embedding rate limiting.

        This asynchronous wrapper checks if the embedding API rate limit is exceeded before
        allowing the function to execute. If the limit is exceeded, it logs a warning and raises
        an EmbeddingException. If not, it waits as necessary and proceeds with the function
        call.

        Parameters:
        -----------

            - *args: Positional arguments passed to the wrapped function.
            - **kwargs: Keyword arguments passed to the wrapped function.

        Returns:
        --------

            Returns the result of the wrapped function after handling rate limiting.
        """
        limiter = EmbeddingRateLimiter.get_instance()

        # Check if rate limiting is enabled and if we're at the limit
        if limiter.hit_limit():
            error_msg = "Embedding API rate limit exceeded"
            logger.warning(error_msg)

            # Create a custom embedding rate limit exception
            from cognee.infrastructure.databases.exceptions import EmbeddingException

            raise EmbeddingException(error_msg)

        # Add this request to the counter and proceed
        await limiter.async_wait_if_needed()
        return await func(*args, **kwargs)

    return wrapper


def embedding_sleep_and_retry_sync(max_retries=5, base_backoff=1.0, jitter=0.5):
    """
    Add retry with exponential backoff for synchronous embedding functions.

    Parameters:
    -----------

        - max_retries: Maximum number of retries before giving up. (default 5)
        - base_backoff: Base backoff time in seconds for retry intervals. (default 1.0)
        - jitter: Jitter factor to randomize the backoff time to avoid collision. (default
          0.5)

    Returns:
    --------

        A decorator that retries the wrapped function on rate limit errors, applying
        exponential backoff with jitter.
    """

    def decorator(func):
        """
        Wraps a function to apply retry logic on rate limit errors.

        Parameters:
        -----------

            - func: The function to be wrapped with retry logic.

        Returns:
        --------

            Returns the wrapped function with retry logic applied.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Retry the execution of a function with backoff on failure due to rate limit errors.

            This wrapper function will call the specified function and if it raises an exception, it
            will handle retries according to defined conditions. It will check the environment for a
            DISABLE_RETRIES flag to determine whether to retry or propagate errors immediately
            during tests. If the error is identified as a rate limit error, it will apply an
            exponential backoff strategy with jitter before retrying, up to a maximum number of
            retries. If the retries are exhausted, it raises the last encountered error.

            Parameters:
            -----------

                - *args: Positional arguments passed to the wrapped function.
                - **kwargs: Keyword arguments passed to the wrapped function.

            Returns:
            --------

                Returns the result of the wrapped function if successful; otherwise, raises the last
                error encountered after maximum retries are exhausted.
            """
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
    Add retry logic with exponential backoff for asynchronous embedding functions.

    This decorator retries the wrapped asynchronous function upon encountering rate limit
    errors, utilizing exponential backoff with optional jitter to space out retry attempts.
    It allows for a maximum number of retries before giving up and raising the last error
    encountered.

    Parameters:
    -----------

        - max_retries: Maximum number of retries allowed before giving up. (default 5)
        - base_backoff: Base amount of time in seconds to wait before retrying after a rate
          limit error. (default 1.0)
        - jitter: Amount of randomness to add to the backoff duration to help mitigate burst
          issues on retries. (default 0.5)

    Returns:
    --------

        Returns a decorated asynchronous function that implements the retry logic on rate
        limit errors.
    """

    def decorator(func):
        """
        Handle retries for an async function with exponential backoff and jitter.

        Parameters:
        -----------

            - func: An asynchronous function to be wrapped with retry logic.

        Returns:
        --------

            Returns the wrapper function that manages the retry behavior for the wrapped async
            function.
        """

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            """
            Handle retries for an async function with exponential backoff and jitter.

            If the environment variable DISABLE_RETRIES is set to true, 1, or yes, the function will
            not retry on errors.
            It attempts to call the wrapped function until it succeeds or the maximum number of
            retries is reached. If an exception occurs, it checks if it's a rate limit error to
            determine if a retry is needed.

            Parameters:
            -----------

                - *args: Positional arguments passed to the wrapped function.
                - **kwargs: Keyword arguments passed to the wrapped function.

            Returns:
            --------

                Returns the result of the wrapped async function if successful; raises the last
                encountered error if all retries fail.
            """
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
