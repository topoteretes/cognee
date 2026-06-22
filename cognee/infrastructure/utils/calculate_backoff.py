import random

# Default retry settings
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0  # exponential backoff multiplier
DEFAULT_JITTER = 0.1  # 10% jitter to avoid thundering herd


def calculate_backoff(
    attempt: int,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    jitter: float = DEFAULT_JITTER,
) -> float:
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
