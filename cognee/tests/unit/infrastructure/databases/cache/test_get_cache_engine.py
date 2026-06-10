"""Tests for the cache engine factory (create_cache_engine backend dispatch)."""

import importlib
import types
from unittest.mock import MagicMock, patch

import pytest

from cognee.infrastructure.databases.exceptions import CacheConnectionError

# The cache package __init__ re-exports the get_cache_engine function, shadowing
# the submodule attribute; import the module itself to reach create_cache_engine.
factory_mod = importlib.import_module("cognee.infrastructure.databases.cache.get_cache_engine")

RELATIONAL_CONFIG_MOD = "cognee.infrastructure.databases.relational.config"
REDIS_ADAPTER_MOD = "cognee.infrastructure.databases.cache.redis.RedisAdapter"


@pytest.fixture(autouse=True)
def clear_factory_cache():
    """create_cache_engine is lru_cache'd; isolate every test case."""
    factory_mod.create_cache_engine.cache_clear()
    yield
    factory_mod.create_cache_engine.cache_clear()


def _fake_cache_config(backend: str):
    return types.SimpleNamespace(caching=True, usage_logging=False, cache_backend=backend)


def _create_engine(backend: str, cache_db_url=None):
    with patch.object(factory_mod, "get_cache_config", return_value=_fake_cache_config(backend)):
        return factory_mod.create_cache_engine(
            cache_host="localhost",
            cache_port=6379,
            cache_username=None,
            cache_password=None,
            lock_key="default_lock",
            log_key="usage_logs",
            cache_db_url=cache_db_url,
        )


def test_sqlite_backend_returns_sql_adapter_with_aiosqlite_url(tmp_path):
    fake_relational = types.SimpleNamespace(db_path=str(tmp_path), db_provider="sqlite")

    with patch(f"{RELATIONAL_CONFIG_MOD}.get_relational_config", return_value=fake_relational):
        engine = _create_engine("sqlite")

    from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter

    assert isinstance(engine, SqlCacheAdapter)
    assert engine.db_uri == f"sqlite+aiosqlite:///{tmp_path}/cache.db"


def test_sqlite_backend_prefers_explicit_cache_db_url(tmp_path):
    explicit_url = f"sqlite+aiosqlite:///{tmp_path}/custom_cache.db"

    engine = _create_engine("sqlite", cache_db_url=explicit_url)

    from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter

    assert isinstance(engine, SqlCacheAdapter)
    assert engine.db_uri == explicit_url


def test_postgres_backend_with_cache_db_url_returns_sql_adapter():
    explicit_url = "postgresql+asyncpg://cognee:cognee@localhost:5432/cognee_db"

    engine = _create_engine("postgres", cache_db_url=explicit_url)

    from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter

    assert isinstance(engine, SqlCacheAdapter)
    assert engine.db_uri == explicit_url


def test_postgres_backend_without_url_or_postgres_relational_raises():
    fake_relational = types.SimpleNamespace(db_provider="sqlite")

    with patch(f"{RELATIONAL_CONFIG_MOD}.get_relational_config", return_value=fake_relational):
        with pytest.raises(CacheConnectionError, match="CACHE_DB_URL or DB_PROVIDER=postgres"):
            _create_engine("postgres")


def test_postgres_backend_falls_back_to_relational_postgres_settings():
    fake_relational = types.SimpleNamespace(
        db_provider="postgres",
        db_username="cognee",
        db_password="cognee",
        db_host="localhost",
        db_port=5432,
        db_name="cognee_db",
    )

    with patch(f"{RELATIONAL_CONFIG_MOD}.get_relational_config", return_value=fake_relational):
        engine = _create_engine("postgres")

    from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter

    assert isinstance(engine, SqlCacheAdapter)
    assert engine.db_uri == "postgresql+asyncpg://cognee:cognee@localhost:5432/cognee_db"


def test_unknown_backend_raises_value_error_listing_all_backends():
    with pytest.raises(ValueError) as exc_info:
        _create_engine("bogus")

    message = str(exc_info.value)
    assert "bogus" in message
    for backend in ("redis", "fs", "tapes", "sqlite", "postgres"):
        assert f"'{backend}'" in message


def test_redis_backend_returns_redis_adapter():
    with (
        patch(f"{REDIS_ADAPTER_MOD}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
        patch(f"{REDIS_ADAPTER_MOD}.aioredis.Redis", return_value=MagicMock()),
    ):
        engine = _create_engine("redis")

        from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

        assert isinstance(engine, RedisAdapter)


def test_caching_disabled_returns_none():
    fake_config = types.SimpleNamespace(caching=False, usage_logging=False, cache_backend="sqlite")

    with patch.object(factory_mod, "get_cache_config", return_value=fake_config):
        engine = factory_mod.create_cache_engine(
            cache_host="localhost",
            cache_port=6379,
            cache_username=None,
            cache_password=None,
            lock_key="default_lock",
            log_key="usage_logs",
        )

    assert engine is None
