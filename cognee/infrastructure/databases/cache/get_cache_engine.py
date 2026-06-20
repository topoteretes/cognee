"""Factory to get the appropriate cache coordination engine (e.g., Redis)."""

import os
from functools import lru_cache
from typing import Optional

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter
from cognee.infrastructure.databases.exceptions import CacheConnectionError

logger = get_logger("CacheEngine")


def _resolve_cache_db_url(backend: str, cache_db_url: Optional[str]) -> str:
    """
    Resolve the SQLAlchemy async URL for the SQL cache backends.

    CACHE_DB_URL wins when set. Otherwise "sqlite" mirrors the relational SQLite
    engine's databases directory (with a dedicated cache.db file), and "postgres"
    falls back to the relational DB_* settings when DB_PROVIDER=postgres.
    """
    if cache_db_url:
        return cache_db_url

    from cognee.infrastructure.databases.relational.config import get_relational_config

    relational_config = get_relational_config()

    if backend == "sqlite":
        db_path = relational_config.db_path
        if "s3://" in db_path:
            raise CacheConnectionError(
                "CACHE_BACKEND=sqlite cannot store cache.db on S3; "
                "set CACHE_DB_URL or CACHE_BACKEND=postgres"
            )
        os.makedirs(db_path, exist_ok=True)
        return f"sqlite+aiosqlite:///{db_path}/cache.db"

    if relational_config.db_provider == "postgres":
        from sqlalchemy import URL

        logger.warning(
            "CACHE_BACKEND=postgres without CACHE_DB_URL; "
            "falling back to the relational DB_* settings."
        )
        return URL.create(
            "postgresql+asyncpg",
            username=relational_config.db_username,
            password=relational_config.db_password,
            host=relational_config.db_host,
            port=int(relational_config.db_port),
            database=relational_config.db_name,
        ).render_as_string(hide_password=False)

    raise CacheConnectionError(
        "CACHE_BACKEND=postgres requires CACHE_DB_URL or DB_PROVIDER=postgres"
    )


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
    cache_db_url: str | None = None,
    cache_purge_interval_seconds: int = 900,
):
    """
    Factory function to instantiate a cache coordination backend.

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
    - cache_db_url: SQLAlchemy async URL for the SQL cache backends.
    - cache_purge_interval_seconds: Minimum interval between global TTL purge sweeps.

    Returns:
    --------
    - CacheDBInterface: An instance of the appropriate cache adapter.
    """
    config = get_cache_config()
    if config.caching or config.usage_logging:
        if config.cache_backend == "redis":
            from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

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
        elif config.cache_backend in ("sqlite", "postgres"):
            from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import (
                SqlCacheAdapter,
            )

            try:
                connection_string = _resolve_cache_db_url(config.cache_backend, cache_db_url)
            except CacheConnectionError as error:
                # An explicitly chosen backend must fail loudly. The implicit
                # sqlite default, however, can land on deployments it cannot
                # serve (e.g. system root on S3) - degrade to the filesystem
                # cache there so sessions keep working out of the box.
                explicitly_chosen = "cache_backend" in getattr(config, "model_fields_set", set())
                if config.cache_backend != "sqlite" or explicitly_chosen:
                    raise
                logger.warning(
                    f"Default sqlite cache backend is unavailable ({error}); "
                    "falling back to the filesystem cache backend. "
                    "Set CACHE_DB_URL or CACHE_BACKEND to override."
                )
                return FSCacheAdapter(session_ttl_seconds=session_ttl_seconds)

            return SqlCacheAdapter(
                connection_string=connection_string,
                lock_key=lock_key,
                log_key=log_key,
                session_ttl_seconds=session_ttl_seconds,
                agentic_lock_expire=agentic_lock_expire,
                agentic_lock_timeout=agentic_lock_timeout,
                purge_interval_seconds=cache_purge_interval_seconds,
            )
        else:
            raise ValueError(
                f"Unsupported cache backend: '{config.cache_backend}'. "
                f"Supported backends are: 'redis', 'fs', 'tapes', 'sqlite', 'postgres'"
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
        cache_db_url=config.cache_db_url,
        cache_purge_interval_seconds=config.cache_purge_interval_seconds,
    )


async def close_cache_engine(
    lock_key: Optional[str] = "default_lock",
    log_key: Optional[str] = "usage_logs",
) -> None:
    """Close and clear the cached cache engine instance."""
    if create_cache_engine.cache_info().currsize == 0:
        return

    try:
        cache_engine = get_cache_engine(lock_key=lock_key, log_key=log_key)
        if cache_engine is not None:
            await cache_engine.close()
    finally:
        create_cache_engine.cache_clear()
