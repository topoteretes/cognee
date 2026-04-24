"""Tests for cache configuration."""

import pytest
from cognee.infrastructure.databases.cache.config import CacheConfig, get_cache_config


def test_cache_config_defaults(monkeypatch):
    """Test that CacheConfig has the correct default values."""
    for env_var in (
        "CACHE_BACKEND",
        "CACHING",
        "AUTO_FEEDBACK",
        "SHARED_KUZU_LOCK",
        "CACHE_HOST",
        "CACHE_PORT",
        "CACHE_USERNAME",
        "CACHE_PASSWORD",
        "AGENTIC_LOCK_EXPIRE",
        "AGENTIC_LOCK_TIMEOUT",
        "SESSION_TTL_SECONDS",
        "USAGE_LOGGING",
        "USAGE_LOGGING_TTL",
    ):
        monkeypatch.delenv(env_var, raising=False)

    config = CacheConfig(_env_file=None)

    assert config.cache_backend == "fs"
    assert config.caching is True
    assert config.shared_kuzu_lock is False
    assert config.cache_host == "localhost"
    assert config.cache_port == 6379
    assert config.agentic_lock_expire == 240
    assert config.agentic_lock_timeout == 300
    assert config.session_ttl_seconds == 604800


def test_cache_config_custom_values():
    """Test that CacheConfig accepts custom values."""
    config = CacheConfig(
        cache_backend="redis",
        caching=True,
        shared_kuzu_lock=True,
        cache_host="redis.example.com",
        cache_port=6380,
        agentic_lock_expire=120,
        agentic_lock_timeout=180,
        session_ttl_seconds=3600,
    )

    assert config.cache_backend == "redis"
    assert config.caching is True
    assert config.shared_kuzu_lock is True
    assert config.cache_host == "redis.example.com"
    assert config.cache_port == 6380
    assert config.agentic_lock_expire == 120
    assert config.agentic_lock_timeout == 180
    assert config.session_ttl_seconds == 3600


def test_cache_config_to_dict():
    """Test the to_dict method returns all configuration values."""
    config = CacheConfig(
        cache_backend="fs",
        caching=True,
        shared_kuzu_lock=True,
        cache_host="test-host",
        cache_port=7000,
        agentic_lock_expire=100,
        agentic_lock_timeout=200,
        session_ttl_seconds=0,
    )

    config_dict = config.to_dict()

    assert config_dict == {
        "cache_backend": "fs",
        "caching": True,
        "auto_feedback": False,
        "shared_kuzu_lock": True,
        "cache_host": "test-host",
        "cache_port": 7000,
        "cache_username": None,
        "cache_password": None,
        "agentic_lock_expire": 100,
        "agentic_lock_timeout": 200,
        "session_ttl_seconds": 0,
        "max_session_context_chars": None,
        "usage_logging": False,
        "usage_logging_ttl": 604800,
        "tapes_ingest_url": "http://localhost:8082",
        "tapes_provider": "openai",
        "tapes_agent_name": "cognee",
        "tapes_model": "cognee-session",
        "tapes_request_timeout": 5.0,
    }


def test_cache_config_session_ttl_none():
    """Test that session_ttl_seconds accepts None to disable Redis session expiry."""
    config = CacheConfig(session_ttl_seconds=None)

    assert config.session_ttl_seconds is None
    assert config.to_dict()["session_ttl_seconds"] is None


def test_get_cache_config_singleton():
    """Test that get_cache_config returns the same instance."""
    config1 = get_cache_config()
    config2 = get_cache_config()

    assert config1 is config2


def test_cache_config_extra_fields_allowed():
    """Test that CacheConfig allows extra fields due to extra='allow'."""
    config = CacheConfig(extra_field="extra_value", another_field=123)

    assert hasattr(config, "extra_field")
    assert config.extra_field == "extra_value"
    assert hasattr(config, "another_field")
    assert config.another_field == 123


def test_cache_config_boolean_type_validation():
    """Test that boolean fields accept various truthy/falsy values."""
    config1 = CacheConfig(caching="true", shared_kuzu_lock="yes")
    assert config1.caching is True
    assert config1.shared_kuzu_lock is True

    config2 = CacheConfig(caching="false", shared_kuzu_lock="no")
    assert config2.caching is False
    assert config2.shared_kuzu_lock is False
