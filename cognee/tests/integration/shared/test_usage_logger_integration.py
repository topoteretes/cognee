"""Integration tests for usage logger with real Redis components."""

import os
import pytest
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID
from unittest.mock import patch

from cognee.shared.usage_logger import log_usage
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.get_cache_engine import (
    get_cache_engine,
    create_cache_engine,
)


@pytest.fixture
def usage_logging_config():
    """Fixture to enable usage logging via environment variables."""
    original_env = os.environ.copy()
    os.environ["USAGE_LOGGING"] = "true"
    os.environ["CACHE_BACKEND"] = "redis"
    os.environ["CACHE_HOST"] = "localhost"
    os.environ["CACHE_PORT"] = "6379"
    get_cache_config.cache_clear()
    create_cache_engine.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(original_env)
    get_cache_config.cache_clear()
    create_cache_engine.cache_clear()


@pytest.fixture
def usage_logging_disabled():
    """Fixture to disable usage logging via environment variables."""
    original_env = os.environ.copy()
    os.environ["USAGE_LOGGING"] = "false"
    os.environ["CACHE_BACKEND"] = "redis"
    get_cache_config.cache_clear()
    create_cache_engine.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(original_env)
    get_cache_config.cache_clear()
    create_cache_engine.cache_clear()


@pytest.fixture
def redis_adapter():
    """Real RedisAdapter instance for testing."""
    from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

    try:
        yield RedisAdapter(host="localhost", port=6379, log_key="test_usage_logs")
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.fixture
def test_user():
    """Test user object."""
    return SimpleNamespace(id="test-user-123")


class TestDecoratorBehavior:
    """Test decorator behavior with real components."""

    @pytest.mark.asyncio
    async def test_decorator_configuration(
        self, usage_logging_disabled, usage_logging_config, redis_adapter
    ):
        """Test decorator skips when disabled and logs when enabled."""
        # Test disabled
        call_count = 0

        @log_usage(function_name="test_func", log_type="test")
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "result"

        assert await test_func() == "result"
        assert call_count == 1

        # Test enabled with cache engine None
        with patch("cognee.shared.usage_logger.get_cache_engine") as mock_get:
            mock_get.return_value = None
            assert await test_func() == "result"

    @pytest.mark.asyncio
    async def test_decorator_logging(self, usage_logging_config, redis_adapter, test_user):
        """Test decorator logs to Redis with correct structure."""

        @log_usage(function_name="test_func", log_type="test")
        async def test_func(param1: str, param2: int = 42, user=None):
            await asyncio.sleep(0.01)
            return {"result": f"{param1}_{param2}"}

        with patch("cognee.shared.usage_logger.get_cache_engine") as mock_get:
            mock_get.return_value = redis_adapter

            result = await test_func("value1", user=test_user)
            assert result == {"result": "value1_42"}

            logs = await redis_adapter.get_usage_logs("test-user-123", limit=10)
            log = logs[0]
            assert log["function_name"] == "test_func"
            assert log["type"] == "test"
            assert log["user_id"] == "test-user-123"
            assert log["parameters"]["param1"] == "value1"
            assert log["parameters"]["param2"] == 42
            assert log["success"] is True
            assert all(
                field in log
                for field in [
                    "timestamp",
                    "result",
                    "error",
                    "duration_ms",
                    "start_time",
                    "end_time",
                    "metadata",
                ]
            )
            assert "cognee_version" in log["metadata"]

    @pytest.mark.asyncio
    async def test_multiple_calls(self, usage_logging_config, redis_adapter, test_user):
        """Test multiple consecutive calls are all logged."""

        @log_usage(function_name="multi_test", log_type="test")
        async def multi_func(call_num: int, user=None):
            return {"call": call_num}

        with patch("cognee.shared.usage_logger.get_cache_engine") as mock_get:
            mock_get.return_value = redis_adapter

            for i in range(3):
                await multi_func(i, user=test_user)

            logs = await redis_adapter.get_usage_logs("test-user-123", limit=10)
            assert len(logs) >= 3
            call_nums = {log["parameters"]["call_num"] for log in logs[:3]}
            assert call_nums == {0, 1, 2}


class TestRealRedisIntegration:
    """Test real Redis integration."""

    @pytest.mark.asyncio
    async def test_redis_storage_retrieval_and_ttl(
        self, usage_logging_config, redis_adapter, test_user
    ):
        """Test logs are stored, retrieved with correct order/limits, and TTL is set."""

        @log_usage(function_name="redis_test", log_type="test")
        async def redis_func(data: str, user=None):
            return {"processed": data}

        @log_usage(function_name="order_test", log_type="test")
        async def order_func(num: int, user=None):
            return {"num": num}

        with patch("cognee.shared.usage_logger.get_cache_engine") as mock_get:
            mock_get.return_value = redis_adapter

            # Storage
            await redis_func("test_data", user=test_user)
            logs = await redis_adapter.get_usage_logs("test-user-123", limit=10)
            assert logs[0]["function_name"] == "redis_test"
            assert logs[0]["parameters"]["data"] == "test_data"

            # Order (most recent first)
            for i in range(3):
                await order_func(i, user=test_user)
                await asyncio.sleep(0.01)

            logs = await redis_adapter.get_usage_logs("test-user-123", limit=10)
            assert [log["parameters"]["num"] for log in logs[:3]] == [2, 1, 0]

            # Limit
            assert len(await redis_adapter.get_usage_logs("test-user-123", limit=2)) == 2

            # TTL
            ttl = await redis_adapter.async_redis.ttl("test_usage_logs:test-user-123")
            assert 0 < ttl <= 604800


class TestEdgeCases:
    """Test edge cases in integration tests."""

    @pytest.mark.asyncio
    async def test_edge_cases(self, usage_logging_config, redis_adapter, test_user):
        """Test no params, defaults, complex structures, exceptions, None, circular refs."""

        @log_usage(function_name="no_params", log_type="test")
        async def no_params_func(user=None):
            return "result"

        @log_usage(function_name="defaults_only", log_type="test")
        async def defaults_only_func(param1: str = "default1", param2: int = 42, user=None):
            return {"param1": param1, "param2": param2}

        @log_usage(function_name="complex_test", log_type="test")
        async def complex_func(user=None):
            return {
                "nested": {
                    "list": [1, 2, 3],
                    "uuid": UUID("123e4567-e89b-12d3-a456-426614174000"),
                    "datetime": datetime(2024, 1, 15, tzinfo=timezone.utc),
                }
            }

        @log_usage(function_name="exception_test", log_type="test")
        async def exception_func(user=None):
            raise RuntimeError("Test exception")

        @log_usage(function_name="none_test", log_type="test")
        async def none_func(user=None):
            return None

        with patch("cognee.shared.usage_logger.get_cache_engine") as mock_get:
            mock_get.return_value = redis_adapter

            # No parameters
            await no_params_func(user=test_user)
            logs = await redis_adapter.get_usage_logs("test-user-123", limit=10)
            assert logs[0]["parameters"] == {}

            # Default parameters
            await defaults_only_func(user=test_user)
            logs = await redis_adapter.get_usage_logs("test-user-123", limit=10)
            assert logs[0]["parameters"]["param1"] == "default1"
            assert logs[0]["parameters"]["param2"] == 42

            # Complex nested structures
            await complex_func(user=test_user)
            logs = await redis_adapter.get_usage_logs("test-user-123", limit=10)
            assert isinstance(logs[0]["result"]["nested"]["uuid"], str)
            assert isinstance(logs[0]["result"]["nested"]["datetime"], str)

            # Exception handling
            with pytest.raises(RuntimeError):
                await exception_func(user=test_user)
            logs = await redis_adapter.get_usage_logs("test-user-123", limit=10)
            assert logs[0]["success"] is False
            assert "Test exception" in logs[0]["error"]

            # None return value
            assert await none_func(user=test_user) is None
            logs = await redis_adapter.get_usage_logs("test-user-123", limit=10)
            assert logs[0]["result"] is None
