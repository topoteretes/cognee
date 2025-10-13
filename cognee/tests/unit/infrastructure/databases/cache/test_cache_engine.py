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
