from functools import lru_cache

from .create_loader_engine import create_loader_engine
from .LoaderEngine import LoaderEngine


@lru_cache
def get_loader_engine() -> LoaderEngine:
    """
    Factory function to get loader engine.

    Follows cognee's pattern with @lru_cache for efficient reuse
    of engine instances. Configuration is loaded from environment
    variables and settings.

    Returns:
        Cached LoaderEngine instance configured with current settings
    """
    return create_loader_engine()
