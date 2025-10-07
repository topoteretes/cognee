"""Factory to get the appropriate cache coordination engine (e.g., Redis)."""

from functools import lru_cache
from cognee.infrastructure.databases.cache.config import get_cache_config

from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface

config = get_cache_config()


@lru_cache
def create_cache_engine(
    cache_host: str,
    cache_port: int,
    lock_key: str,
    agentic_lock_expire: int = 240,
    agentic_lock_timeout: int = 300,
):
    """
    Factory function to instantiate a cache coordination backend (currently Redis).

    Parameters:
    -----------
    - cache_host: Hostname or IP of the cache server.
    - cache_port: Port number to connect to.
    - lock_key: Identifier used for the locking resource.
    - agentic_lock_expire: Duration to hold the lock after acquisition.
    - agentic_lock_timeout: Max time to wait for the lock before failing.

    Returns:
    --------
    - CacheDBInterface: An instance of the appropriate cache adapter. :TODO: Now we support only Redis. later if we add more here we can split the logic
    """
    if config.caching:
        from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

        return RedisAdapter(
            host=cache_host,
            port=cache_port,
            lock_name=lock_key,
            timeout=agentic_lock_expire,
            blocking_timeout=agentic_lock_timeout,
        )
    else:
        return None


def get_cache_engine() -> CacheDBInterface:
    """
    Returns a cache adapter instance using current context configuration.
    """

    return create_cache_engine(
        cache_host=config.cache_host,
        cache_port=config.cache_port,
        lock_key=config.lock_key,
        agentic_lock_expire=config.agentic_lock_expire,
        agentic_lock_timeout=config.agentic_lock_timeout,
    )
