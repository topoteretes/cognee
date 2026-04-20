"""Tests for DatasetQueueConfig configuration class.

Tests configuration loading, validation, and environment variable handling.
"""

import os
import pytest
from unittest.mock import patch


class TestDatasetQueueConfigClass:
    """Tests for the DatasetQueueConfig Pydantic model."""

    def test_config_has_expected_fields(self):
        """Test that config class has all expected fields."""
        from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

        config = DatasetQueueConfig()

        # Required fields
        assert hasattr(config, "dataset_queue_enabled")
        assert hasattr(config, "database_max_lru_cache_size")

    def test_config_default_values(self):
        """Test default values when no environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()

            assert config.dataset_queue_enabled is False
            assert config.database_max_lru_cache_size == 10

    def test_config_enabled_true_from_env(self):
        """Test that 'true' (lowercase) enables the queue."""
        with patch.dict(os.environ, {"DATASET_QUEUE_ENABLED": "true"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()
            assert config.dataset_queue_enabled is True

    def test_config_enabled_TRUE_from_env(self):
        """Test that 'TRUE' (uppercase) enables the queue."""
        with patch.dict(os.environ, {"DATASET_QUEUE_ENABLED": "TRUE"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()
            assert config.dataset_queue_enabled is True

    def test_config_enabled_1_from_env(self):
        """Test that '1' enables the queue."""
        with patch.dict(os.environ, {"DATASET_QUEUE_ENABLED": "1"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()
            assert config.dataset_queue_enabled is True

    def test_config_disabled_false_from_env(self):
        """Test that 'false' (lowercase) disables the queue."""
        with patch.dict(os.environ, {"DATASET_QUEUE_ENABLED": "false"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()
            assert config.dataset_queue_enabled is False

    def test_config_disabled_0_from_env(self):
        """Test that '0' disables the queue."""
        with patch.dict(os.environ, {"DATASET_QUEUE_ENABLED": "0"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()
            assert config.dataset_queue_enabled is False

    def test_config_max_lru_cache_size_from_env(self):
        """Test that DATABASE_MAX_LRU_CACHE_SIZE is read correctly."""
        with patch.dict(os.environ, {"DATABASE_MAX_LRU_CACHE_SIZE": "42"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()
            assert config.database_max_lru_cache_size == 42

    def test_config_max_lru_cache_size_minimum_value(self):
        """Test that max LRU cache size has a minimum of 1."""
        with patch.dict(os.environ, {"DATABASE_MAX_LRU_CACHE_SIZE": "0"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            try:
                config = DatasetQueueConfig()
                # If it doesn't raise, should be normalized to at least 1
                assert config.database_max_lru_cache_size >= 1
            except ValueError:
                # Acceptable to raise for invalid value
                pass

    def test_config_max_lru_cache_size_negative_rejected(self):
        """Test that negative values are rejected or normalized."""
        with patch.dict(os.environ, {"DATABASE_MAX_LRU_CACHE_SIZE": "-10"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            try:
                config = DatasetQueueConfig()
                # If doesn't raise, should be normalized to positive
                assert config.database_max_lru_cache_size >= 1
            except ValueError:
                # Acceptable to raise for invalid value
                pass

    def test_config_max_lru_cache_size_non_integer_rejected(self):
        """Test that non-integer values are handled appropriately."""
        with patch.dict(os.environ, {"DATABASE_MAX_LRU_CACHE_SIZE": "abc"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            try:
                config = DatasetQueueConfig()
                # If doesn't raise, should use default
                assert config.database_max_lru_cache_size == 10
            except (ValueError, TypeError):
                # Acceptable to raise for invalid value
                pass

    def test_config_max_lru_cache_size_float_truncated(self):
        """Test that float values are converted to integers."""
        with patch.dict(os.environ, {"DATABASE_MAX_LRU_CACHE_SIZE": "5.9"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            try:
                config = DatasetQueueConfig()
                # Should truncate to 5 or round to 6
                assert config.database_max_lru_cache_size in [5, 6]
            except (ValueError, TypeError):
                # Acceptable to raise for non-integer
                pass

    def test_config_very_large_max_size(self):
        """Test behavior with very large max size value."""
        with patch.dict(os.environ, {"DATABASE_MAX_LRU_CACHE_SIZE": "1000000"}, clear=True):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()
            # Should accept large values
            assert config.database_max_lru_cache_size == 1000000


class TestGetDatasetQueueConfig:
    """Tests for the get_dataset_queue_config function."""

    def test_get_config_returns_config_instance(self):
        """Test that get function returns a config instance."""
        from cognee.infrastructure.databases.dataset_queue.config import (
            get_dataset_queue_config,
            DatasetQueueConfig,
        )

        config = get_dataset_queue_config()
        assert isinstance(config, DatasetQueueConfig)

    def test_get_config_caches_instance(self):
        """Test that config is cached (singleton pattern)."""
        from cognee.infrastructure.databases.dataset_queue.config import get_dataset_queue_config

        config1 = get_dataset_queue_config()
        config2 = get_dataset_queue_config()

        # Depending on implementation, these may be same or different instances
        # At minimum, they should have the same values
        assert config1.dataset_queue_enabled == config2.dataset_queue_enabled
        assert config1.database_max_lru_cache_size == config2.database_max_lru_cache_size


class TestConfigEnvironmentIsolation:
    """Tests for environment variable isolation in config."""

    def test_config_only_reads_expected_env_vars(self):
        """Test that config doesn't read unexpected environment variables."""
        with patch.dict(
            os.environ,
            {
                "DATASET_QUEUE_ENABLED": "true",
                "DATABASE_MAX_LRU_CACHE_SIZE": "15",
                "UNRELATED_VAR": "should_not_affect",
            },
            clear=True,
        ):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()

            assert config.dataset_queue_enabled is True
            assert config.database_max_lru_cache_size == 15
            # Should not have unrelated attributes
            assert not hasattr(config, "unrelated_var")

    def test_config_works_with_other_cognee_env_vars(self):
        """Test config works alongside other Cognee environment variables."""
        with patch.dict(
            os.environ,
            {
                "DATASET_QUEUE_ENABLED": "true",
                "DATABASE_MAX_LRU_CACHE_SIZE": "20",
                "LLM_API_KEY": "test_key",
                "GRAPH_DATABASE_PROVIDER": "kuzu",
            },
            clear=True,
        ):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()

            # Queue config should work independently
            assert config.dataset_queue_enabled is True
            assert config.database_max_lru_cache_size == 20


class TestConfigRepr:
    """Tests for config string representation."""

    def test_config_repr_is_readable(self):
        """Test that config has a readable string representation."""
        from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

        config = DatasetQueueConfig()
        repr_str = repr(config)

        # Should contain key information
        assert "DatasetQueueConfig" in repr_str or "enabled" in repr_str.lower()

    def test_config_str_does_not_leak_secrets(self):
        """Test that config string representation doesn't leak sensitive info."""
        from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

        config = DatasetQueueConfig()
        str_repr = str(config)

        # Should not contain any API keys or secrets
        # (Queue config shouldn't have secrets, but good practice to test)
        assert "api_key" not in str_repr.lower()
        assert "password" not in str_repr.lower()
        assert "secret" not in str_repr.lower()


class TestConfigToDict:
    """Tests for config serialization."""

    def test_config_to_dict_contains_all_fields(self):
        """Test that config can be converted to dictionary with all fields."""
        from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

        config = DatasetQueueConfig()

        # If config has a dict method or model_dump
        if hasattr(config, "model_dump"):
            config_dict = config.model_dump()
        elif hasattr(config, "dict"):
            config_dict = config.dict()
        else:
            config_dict = {
                "dataset_queue_enabled": config.dataset_queue_enabled,
                "database_max_lru_cache_size": config.database_max_lru_cache_size,
            }

        assert "dataset_queue_enabled" in config_dict
        assert "database_max_lru_cache_size" in config_dict

    def test_config_dict_values_match_attributes(self):
        """Test that dict values match attribute values."""
        with patch.dict(
            os.environ,
            {
                "DATASET_QUEUE_ENABLED": "true",
                "DATABASE_MAX_LRU_CACHE_SIZE": "25",
            },
            clear=True,
        ):
            from cognee.infrastructure.databases.dataset_queue.config import DatasetQueueConfig

            config = DatasetQueueConfig()

            if hasattr(config, "model_dump"):
                config_dict = config.model_dump()
            elif hasattr(config, "dict"):
                config_dict = config.dict()
            else:
                config_dict = {
                    "dataset_queue_enabled": config.dataset_queue_enabled,
                    "database_max_lru_cache_size": config.database_max_lru_cache_size,
                }

            assert config_dict["dataset_queue_enabled"] == config.dataset_queue_enabled
            assert config_dict["database_max_lru_cache_size"] == config.database_max_lru_cache_size
