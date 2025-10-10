"""Tests for cache engine factory methods."""

import pytest
from unittest.mock import patch, MagicMock
from cognee.infrastructure.databases.cache.get_cache_engine import (
    create_cache_engine,
    get_cache_engine,
)
from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter


@pytest.fixture(autouse=True)
def reset_factory_cache():
    """Reset the lru_cache between tests."""
    create_cache_engine.cache_clear()
    yield
    create_cache_engine.cache_clear()


@pytest.fixture
def mock_cache_config():
    """Fixture to mock cache configuration."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.get_cache_config") as mock:
        mock_config = MagicMock()
        mock_config.caching = True
        mock_config.cache_host = "localhost"
        mock_config.cache_port = 6379
        mock_config.agentic_lock_expire = 240
        mock_config.agentic_lock_timeout = 300
        mock.return_value = mock_config
        yield mock_config


def test_create_cache_engine_with_caching_enabled(mock_cache_config):
    """Test that create_cache_engine returns RedisAdapter when caching is enabled."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.config") as mock_config:
        mock_config.caching = True

        with patch("redis.Redis"):
            engine = create_cache_engine(
                cache_host="localhost",
                cache_port=6379,
                lock_key="test-lock",
                agentic_lock_expire=240,
                agentic_lock_timeout=300,
            )

            assert engine is not None
            assert isinstance(engine, RedisAdapter)
            assert engine.host == "localhost"
            assert engine.port == 6379
            assert engine.lock_key == "test-lock"
            assert engine.timeout == 240
            assert engine.blocking_timeout == 300


def test_create_cache_engine_with_caching_disabled():
    """Test that create_cache_engine returns None when caching is disabled."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.config") as mock_config:
        mock_config.caching = False

        engine = create_cache_engine(
            cache_host="localhost",
            cache_port=6379,
            lock_key="test-lock",
        )

        assert engine is None


def test_create_cache_engine_caching():
    """Test that create_cache_engine uses lru_cache and returns same instance."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.config") as mock_config:
        mock_config.caching = True

        with patch("redis.Redis"):
            engine1 = create_cache_engine(
                cache_host="localhost",
                cache_port=6379,
                lock_key="test-lock",
            )

            engine2 = create_cache_engine(
                cache_host="localhost",
                cache_port=6379,
                lock_key="test-lock",
            )

            assert engine1 is engine2


def test_create_cache_engine_different_params():
    """Test that create_cache_engine returns different instances for different parameters."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.config") as mock_config:
        mock_config.caching = True

        with patch("redis.Redis"):
            engine1 = create_cache_engine(
                cache_host="localhost",
                cache_port=6379,
                lock_key="lock-1",
            )

            engine2 = create_cache_engine(
                cache_host="localhost",
                cache_port=6379,
                lock_key="lock-2",
            )

            assert engine1 is not engine2
            assert engine1.lock_key == "lock-1"
            assert engine2.lock_key == "lock-2"


def test_create_cache_engine_custom_timeouts():
    """Test that create_cache_engine accepts custom timeout values."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.config") as mock_config:
        mock_config.caching = True

        with patch("redis.Redis"):
            engine = create_cache_engine(
                cache_host="redis.example.com",
                cache_port=6380,
                lock_key="custom-lock",
                agentic_lock_expire=120,
                agentic_lock_timeout=150,
            )

            assert isinstance(engine, RedisAdapter)
            assert engine.host == "redis.example.com"
            assert engine.port == 6380
            assert engine.timeout == 120
            assert engine.blocking_timeout == 150


def test_get_cache_engine_uses_config(mock_cache_config):
    """Test that get_cache_engine uses configuration values."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.config", mock_cache_config):
        mock_cache_config.caching = True

        with patch("redis.Redis"):
            with patch(
                "cognee.infrastructure.databases.cache.get_cache_engine.create_cache_engine"
            ) as mock_create:
                mock_create.return_value = MagicMock()

                get_cache_engine("my-lock-key")

                mock_create.assert_called_once_with(
                    cache_host="localhost",
                    cache_port=6379,
                    lock_key="my-lock-key",
                    agentic_lock_expire=240,
                    agentic_lock_timeout=300,
                )


def test_get_cache_engine_with_custom_config():
    """Test that get_cache_engine properly uses custom config values."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.config") as mock_config:
        mock_config.caching = True
        mock_config.cache_host = "custom-redis"
        mock_config.cache_port = 7000
        mock_config.agentic_lock_expire = 100
        mock_config.agentic_lock_timeout = 200

        with patch("redis.Redis"):
            engine = get_cache_engine("test-key")

            assert isinstance(engine, RedisAdapter)
            assert engine.host == "custom-redis"
            assert engine.port == 7000
            assert engine.lock_key == "test-key"
            assert engine.timeout == 100
            assert engine.blocking_timeout == 200


def test_get_cache_engine_when_disabled():
    """Test that get_cache_engine returns None when caching is disabled."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.config") as mock_config:
        mock_config.caching = False

        engine = get_cache_engine("test-key")

        assert engine is None


def test_create_cache_engine_default_timeout_values():
    """Test that create_cache_engine uses default timeout values."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.config") as mock_config:
        mock_config.caching = True

        with patch("redis.Redis"):
            engine = create_cache_engine(
                cache_host="localhost",
                cache_port=6379,
                lock_key="test-lock",
            )

            assert isinstance(engine, RedisAdapter)
            assert engine.timeout == 240
            assert engine.blocking_timeout == 300


def test_full_workflow_with_context_manager():
    """Test complete workflow: config -> factory -> adapter with context manager."""
    with patch("cognee.infrastructure.databases.cache.get_cache_engine.config") as mock_config:
        mock_config.caching = True
        mock_config.cache_host = "localhost"
        mock_config.cache_port = 6379
        mock_config.agentic_lock_expire = 240
        mock_config.agentic_lock_timeout = 300

        with patch("redis.Redis") as mock_redis_class:
            mock_redis_instance = MagicMock()
            mock_redis_class.return_value = mock_redis_instance

            mock_lock = MagicMock()
            mock_lock.acquire.return_value = True
            mock_redis_instance.lock.return_value = mock_lock

            engine = get_cache_engine("integration-test-lock")

            assert isinstance(engine, RedisAdapter)
            with engine.hold():
                mock_lock.acquire.assert_called_once()

            mock_lock.release.assert_called_once()
