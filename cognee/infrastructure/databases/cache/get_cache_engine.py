"""Factory to get the appropriate cache coordination engine (e.g., Redis)."""

from functools import lru_cache
from typing import Optional
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter

config = get_cache_config()


@lru_cache
def create_cache_engine(
    cache_host: str,
    cache_port: int,
    cache_username: str,
    cache_password: str,
    lock_key: str,
    log_key: str,
    agentic_lock_expire: int = 240,
    agentic_lock_timeout: int = 300,
):
    """
    Factory function to instantiate a cache coordination backend (currently Redis).

    Parameters:
    -----------
    - cache_host: Hostname or IP of the cache server.
    - cache_port: Port number to connect to.
    - cache_username: Username to authenticate with.
    - cache_password: Password to authenticate with.
    - lock_key: Identifier used for the locking resource.
    - log_key: Identifier used for usage logging.
    - agentic_lock_expire: Duration to hold the lock after acquisition.
    - agentic_lock_timeout: Max time to wait for the lock before failing.

    Returns:
    --------
    - CacheDBInterface: An instance of the appropriate cache adapter.
    """
    if config.caching or config.usage_logging:
        from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

        if config.cache_backend == "redis":
            return RedisAdapter(
                host=cache_host,
                port=cache_port,
                username=cache_username,
                password=cache_password,
                lock_name=lock_key,
                log_key=log_key,
                timeout=agentic_lock_expire,
                blocking_timeout=agentic_lock_timeout,
            )
        elif config.cache_backend == "fs":
            return FSCacheAdapter()
        else:
            raise ValueError(
                f"Unsupported cache backend: '{config.cache_backend}'. "
                f"Supported backends are: 'redis', 'fs'"
            )
    else:
        return None


def get_cache_engine(
    lock_key: Optional[str] = "default_lock",
    log_key: Optional[str] = "usage_logs",
) -> Optional[CacheDBInterface]:
    """
    Returns a cache adapter instance using current context configuration.
    """

    return create_cache_engine(
        cache_host=config.cache_host,
        cache_port=config.cache_port,
        cache_username=config.cache_username,
        cache_password=config.cache_password,
        lock_key=lock_key,
        log_key=log_key,
        agentic_lock_expire=config.agentic_lock_expire,
        agentic_lock_timeout=config.agentic_lock_timeout,
    )
