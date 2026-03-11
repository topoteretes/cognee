import os
from unittest.mock import patch
from cognee.infrastructure.databases.relational.config import RelationalConfig


class TestRelationalConfig:
    """Test suite for RelationalConfig DATABASE_CONNECT_ARGS parsing."""

    def test_database_connect_args_valid_json_dict(self):
        """Test that DATABASE_CONNECT_ARGS is parsed correctly when it's a valid JSON dict."""
        with patch.dict(
            os.environ, {"DATABASE_CONNECT_ARGS": '{"timeout": 60, "sslmode": "require"}'}
        ):
            config = RelationalConfig()
            assert config.database_connect_args == {"timeout": 60, "sslmode": "require"}

    def test_database_connect_args_empty_string(self):
        """Test that empty DATABASE_CONNECT_ARGS is handled correctly."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": ""}):
            config = RelationalConfig()
            assert config.database_connect_args == ""

    def test_database_connect_args_not_set(self):
        """Test that missing DATABASE_CONNECT_ARGS results in None."""
        with patch.dict(os.environ, {}, clear=True):
            config = RelationalConfig()
            assert config.database_connect_args is None

    def test_database_connect_args_invalid_json(self):
        """Test that invalid JSON in DATABASE_CONNECT_ARGS results in empty dict."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"timeout": 60'}):  # Invalid JSON
            config = RelationalConfig()
            assert config.database_connect_args == {}

    def test_database_connect_args_non_dict_json(self):
        """Test that non-dict JSON in DATABASE_CONNECT_ARGS results in empty dict."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '["list", "instead", "of", "dict"]'}):
            config = RelationalConfig()
            assert config.database_connect_args == {}

    def test_database_connect_args_to_dict(self):
        """Test that database_connect_args is included in to_dict() output."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"timeout": 60}'}):
            config = RelationalConfig()
            config_dict = config.to_dict()
            assert "database_connect_args" in config_dict
            assert config_dict["database_connect_args"] == {"timeout": 60}

    def test_database_connect_args_integer_value(self):
        """Test that DATABASE_CONNECT_ARGS with integer values is parsed correctly."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"connect_timeout": 10}'}):
            config = RelationalConfig()
            assert config.database_connect_args == {"connect_timeout": 10}

    def test_database_connect_args_mixed_types(self):
        """Test that DATABASE_CONNECT_ARGS with mixed value types is parsed correctly."""
        with patch.dict(
            os.environ,
            {
                "DATABASE_CONNECT_ARGS": '{"timeout": 60, "sslmode": "require", "retries": 3, "keepalive": true}'
            },
        ):
            config = RelationalConfig()
            assert config.database_connect_args == {
                "timeout": 60,
                "sslmode": "require",
                "retries": 3,
                "keepalive": True,
            }


class TestRelationalConfigSSLMode:
    """Test suite for RelationalConfig DB_SSL_MODE handling."""

    def test_ssl_mode_not_set(self):
        """When DB_SSL_MODE is absent, db_ssl_mode should be None."""
        with patch.dict(os.environ, {}, clear=True):
            config = RelationalConfig()
            assert config.db_ssl_mode is None

    def test_ssl_mode_set_require(self):
        """When DB_SSL_MODE=require, it should be read correctly."""
        with patch.dict(os.environ, {"DB_SSL_MODE": "require"}):
            config = RelationalConfig()
            assert config.db_ssl_mode == "require"

    def test_ssl_mode_set_verify_full(self):
        """When DB_SSL_MODE=verify-full, it should be read correctly."""
        with patch.dict(os.environ, {"DB_SSL_MODE": "verify-full"}):
            config = RelationalConfig()
            assert config.db_ssl_mode == "verify-full"

    def test_ssl_mode_included_in_to_dict(self):
        """db_ssl_mode should be present in to_dict() output."""
        with patch.dict(os.environ, {"DB_SSL_MODE": "prefer"}):
            config = RelationalConfig()
            config_dict = config.to_dict()
            assert "db_ssl_mode" in config_dict
            assert config_dict["db_ssl_mode"] == "prefer"

    def test_ssl_mode_none_in_to_dict(self):
        """When DB_SSL_MODE is absent, to_dict() should include db_ssl_mode as None."""
        with patch.dict(os.environ, {}, clear=True):
            config = RelationalConfig()
            config_dict = config.to_dict()
            assert "db_ssl_mode" in config_dict
            assert config_dict["db_ssl_mode"] is None
