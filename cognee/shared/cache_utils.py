"""Cache utilities for cognee."""

import os
from functools import lru_cache


def cacheable(func=None, *, key=None):
    """
    Decorator to optionally cache function results based on environment variables.

    Usage:
        @cacheable
        def my_func(): ...

        @cacheable(key="VECTOR_ENGINE")
        def my_func(): ...

    Configuration:
        COGNEE_DISABLE_ALL_CACHES=true: Disables caching for all functions decorated with @cacheable
        COGNEE_DISABLE_{KEY}_CACHE=true: Disables caching for specific function if key is provided

    Example:
        # Disable all caches globally
        COGNEE_DISABLE_ALL_CACHES=true

        # Disable only vector engine cache
        COGNEE_DISABLE_VECTOR_ENGINE_CACHE=true
    """

    def decorator(f):
        # Check global kill switch first
        if os.getenv("COGNEE_DISABLE_ALL_CACHES", "false").lower() == "true":
            return f

        # Check specific feature kill switch if key is provided
        if key and os.getenv(f"COGNEE_DISABLE_{key}_CACHE", "false").lower() == "true":
            return f

        return lru_cache(f)

    if func is None:
        # Called with arguments: @cacheable(key="SOMETHING")
        return decorator
    else:
        # Called without arguments: @cacheable
        return decorator(func)
