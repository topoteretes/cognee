"""Factory to get the appropriate cache coordination engine (e.g., Redis)."""

from functools import lru_cache
from typing import Optional
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter


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
    session_ttl_seconds: int | None = 604800,
    tapes_ingest_url: str = "http://localhost:8082",
    tapes_provider: str = "openai",
    tapes_agent_name: str = "cognee",
    tapes_model: str = "cognee-session",
    tapes_request_timeout: float = 5.0,
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
    config = get_cache_config()
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
                session_ttl_seconds=session_ttl_seconds,
            )
        elif config.cache_backend == "fs":
            return FSCacheAdapter(session_ttl_seconds=session_ttl_seconds)
        elif config.cache_backend == "tapes":
            from cognee.infrastructure.databases.cache.tapes.TapesCacheAdapter import (
                TapesCacheAdapter,
            )

            return TapesCacheAdapter(
                session_ttl_seconds=session_ttl_seconds,
                tapes_ingest_url=tapes_ingest_url,
                tapes_provider=tapes_provider,
                tapes_agent_name=tapes_agent_name,
                tapes_model=tapes_model,
                tapes_request_timeout=tapes_request_timeout,
            )
        else:
            raise ValueError(
                f"Unsupported cache backend: '{config.cache_backend}'. "
                f"Supported backends are: 'redis', 'fs', 'tapes'"
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
    config = get_cache_config()

    return create_cache_engine(
        cache_host=config.cache_host,
        cache_port=config.cache_port,
        cache_username=config.cache_username,
        cache_password=config.cache_password,
        lock_key=lock_key,
        log_key=log_key,
        agentic_lock_expire=config.agentic_lock_expire,
        agentic_lock_timeout=config.agentic_lock_timeout,
        session_ttl_seconds=config.session_ttl_seconds,
        tapes_ingest_url=config.tapes_ingest_url,
        tapes_provider=config.tapes_provider,
        tapes_agent_name=config.tapes_agent_name,
        tapes_model=config.tapes_model,
        tapes_request_timeout=config.tapes_request_timeout,
    )
