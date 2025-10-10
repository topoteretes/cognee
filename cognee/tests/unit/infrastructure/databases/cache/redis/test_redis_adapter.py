"""Tests for the RedisAdapter class."""

import pytest
from unittest.mock import MagicMock, patch
import redis
from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter
from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface


@pytest.fixture
def mock_redis():
    """Fixture to mock redis.Redis client."""
    with patch("cognee.infrastructure.databases.cache.redis.RedisAdapter.redis.Redis") as mock:
        yield mock


@pytest.fixture
def redis_adapter(mock_redis):
    """Fixture to create a RedisAdapter instance with mocked Redis client."""
    adapter = RedisAdapter(
        host="localhost",
        port=6379,
        lock_name="test-lock",
        timeout=240,
        blocking_timeout=300,
    )
    return adapter


def test_redis_adapter_initialization(mock_redis):
    """Test that RedisAdapter initializes correctly."""
    adapter = RedisAdapter(
        host="localhost",
        port=6379,
        lock_name="my-lock",
        timeout=120,
        blocking_timeout=150,
    )

    assert adapter.host == "localhost"
    assert adapter.port == 6379
    assert adapter.lock_key == "my-lock"
    assert adapter.timeout == 120
    assert adapter.blocking_timeout == 150
    assert adapter.lock is None

    mock_redis.assert_called_once_with(host="localhost", port=6379)


def test_redis_adapter_inherits_cache_db_interface(mock_redis):
    """Test that RedisAdapter properly inherits from CacheDBInterface."""
    adapter = RedisAdapter(
        host="localhost",
        port=6379,
        lock_name="test-lock",
    )

    assert isinstance(adapter, CacheDBInterface)


def test_redis_adapter_custom_parameters(mock_redis):
    """Test RedisAdapter with custom parameters."""
    adapter = RedisAdapter(
        host="redis.example.com",
        port=6380,
        lock_name="custom-lock-key",
        timeout=60,
        blocking_timeout=90,
    )

    assert adapter.host == "redis.example.com"
    assert adapter.port == 6380
    assert adapter.lock_key == "custom-lock-key"
    assert adapter.timeout == 60
    assert adapter.blocking_timeout == 90


def test_acquire_lock_success(redis_adapter):
    """Test successful lock acquisition."""
    mock_lock = MagicMock()
    mock_lock.acquire.return_value = True
    redis_adapter.redis.lock.return_value = mock_lock

    result = redis_adapter.acquire()

    assert result is mock_lock
    assert redis_adapter.lock is mock_lock
    redis_adapter.redis.lock.assert_called_once_with(
        name="test-lock",
        timeout=240,
        blocking_timeout=300,
    )
    mock_lock.acquire.assert_called_once()


def test_acquire_lock_failure(redis_adapter):
    """Test lock acquisition failure raises RuntimeError."""
    mock_lock = MagicMock()
    mock_lock.acquire.return_value = False
    redis_adapter.redis.lock.return_value = mock_lock

    with pytest.raises(RuntimeError, match="Could not acquire Redis lock: test-lock"):
        redis_adapter.acquire()

    redis_adapter.redis.lock.assert_called_once()
    mock_lock.acquire.assert_called_once()


def test_release_lock_success(redis_adapter):
    """Test successful lock release."""
    mock_lock = MagicMock()
    redis_adapter.lock = mock_lock

    redis_adapter.release()

    mock_lock.release.assert_called_once()
    assert redis_adapter.lock is None


def test_release_lock_when_none(redis_adapter):
    """Test releasing lock when no lock is held."""
    redis_adapter.lock = None

    redis_adapter.release()


def test_release_lock_handles_lock_error(redis_adapter):
    """Test that release handles redis.exceptions.LockError gracefully."""
    mock_lock = MagicMock()
    mock_lock.release.side_effect = redis.exceptions.LockError("Lock not owned")
    redis_adapter.lock = mock_lock

    redis_adapter.release()

    mock_lock.release.assert_called_once()


def test_release_lock_propagates_other_exceptions(redis_adapter):
    """Test that release propagates non-LockError exceptions."""
    mock_lock = MagicMock()
    mock_lock.release.side_effect = redis.exceptions.ConnectionError("Connection lost")
    redis_adapter.lock = mock_lock

    with pytest.raises(redis.exceptions.ConnectionError):
        redis_adapter.release()


def test_hold_context_manager_success(redis_adapter):
    """Test hold context manager with successful acquisition and release."""
    mock_lock = MagicMock()
    mock_lock.acquire.return_value = True
    redis_adapter.redis.lock.return_value = mock_lock

    with redis_adapter.hold():
        assert redis_adapter.lock is mock_lock
        mock_lock.acquire.assert_called_once()

    mock_lock.release.assert_called_once()


def test_hold_context_manager_with_exception(redis_adapter):
    """Test that hold context manager releases lock even when exception occurs."""
    mock_lock = MagicMock()
    mock_lock.acquire.return_value = True
    redis_adapter.redis.lock.return_value = mock_lock

    with pytest.raises(ValueError):
        with redis_adapter.hold():
            mock_lock.acquire.assert_called_once()
            raise ValueError("Test exception")

    mock_lock.release.assert_called_once()


def test_hold_context_manager_acquire_failure(redis_adapter):
    """Test hold context manager when lock acquisition fails."""
    mock_lock = MagicMock()
    mock_lock.acquire.return_value = False
    redis_adapter.redis.lock.return_value = mock_lock

    with pytest.raises(RuntimeError, match="Could not acquire Redis lock"):
        with redis_adapter.hold():
            pass

    mock_lock.release.assert_not_called()


def test_multiple_acquire_release_cycles(redis_adapter):
    """Test multiple acquire/release cycles."""
    mock_lock1 = MagicMock()
    mock_lock1.acquire.return_value = True
    mock_lock2 = MagicMock()
    mock_lock2.acquire.return_value = True

    redis_adapter.redis.lock.side_effect = [mock_lock1, mock_lock2]

    redis_adapter.acquire()
    assert redis_adapter.lock is mock_lock1
    redis_adapter.release()
    assert redis_adapter.lock is None

    redis_adapter.acquire()
    assert redis_adapter.lock is mock_lock2
    redis_adapter.release()
    assert redis_adapter.lock is None


def test_redis_adapter_redis_client_initialization():
    """Test that Redis client is initialized with correct connection parameters."""
    with patch(
        "cognee.infrastructure.databases.cache.redis.RedisAdapter.redis.Redis"
    ) as mock_redis_class:
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance

        adapter = RedisAdapter(
            host="redis-server.example.com",
            port=6380,
            lock_name="test-lock",
        )

        mock_redis_class.assert_called_once_with(
            host="redis-server.example.com",
            port=6380,
        )
        assert adapter.redis is mock_redis_instance


def test_lock_name_vs_lock_key_parameter():
    """Test that lock_name parameter is correctly assigned to lock_key attribute."""
    with patch("cognee.infrastructure.databases.cache.redis.RedisAdapter.redis.Redis"):
        adapter = RedisAdapter(
            host="localhost",
            port=6379,
            lock_name="my-custom-lock-name",
        )

        assert adapter.lock_key == "my-custom-lock-name"


def test_default_timeout_parameters():
    """Test default timeout parameters."""
    with patch("cognee.infrastructure.databases.cache.redis.RedisAdapter.redis.Redis"):
        adapter = RedisAdapter(
            host="localhost",
            port=6379,
            lock_name="test-lock",
        )

        assert adapter.timeout == 240
        assert adapter.blocking_timeout == 300


def test_release_clears_lock_reference(redis_adapter):
    """Test that release clears the lock reference."""
    mock_lock = MagicMock()
    redis_adapter.lock = mock_lock

    redis_adapter.release()

    assert redis_adapter.lock is None
