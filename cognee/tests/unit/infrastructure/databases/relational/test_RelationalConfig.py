import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError
from cognee.infrastructure.databases.relational.config import RelationalConfig


class TestRelationalConfig:
    """Test suite for RelationalConfig DATABASE_CONNECT_ARGS parsing."""

    def test_database_connect_args_valid_json_dict(self):
        """Test that DATABASE_CONNECT_ARGS is parsed correctly when it's a valid JSON dict."""
        with patch.dict(
            os.environ, {"DATABASE_CONNECT_ARGS": '{"timeout": 60, "sslmode": "require"}'}
        ):
            config = RelationalConfig()
            assert config.database_connect_args == (("sslmode", "require"), ("timeout", 60))

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
        """Test that invalid JSON in DATABASE_CONNECT_ARGS raises ValidationError."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"timeout": 60'}):
            with pytest.raises(ValidationError):
                RelationalConfig()

    def test_database_connect_args_non_dict_json(self):
        """Test that non-dict JSON in DATABASE_CONNECT_ARGS raises ValidationError."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '["list", "instead", "of", "dict"]'}):
            with pytest.raises(ValidationError):
                RelationalConfig()

    def test_database_connect_args_to_dict(self):
        """Test that database_connect_args is included in to_dict() output."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"timeout": 60}'}):
            config = RelationalConfig()
            config_dict = config.to_dict()
            assert "database_connect_args" in config_dict
            assert config_dict["database_connect_args"] == (("timeout", 60),)

    def test_database_connect_args_integer_value(self):
        """Test that DATABASE_CONNECT_ARGS with integer values is parsed correctly."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"connect_timeout": 10}'}):
            config = RelationalConfig()
            assert config.database_connect_args == (("connect_timeout", 10),)

    def test_database_connect_args_mixed_types(self):
        """Test that DATABASE_CONNECT_ARGS with mixed value types is parsed correctly."""
        with patch.dict(
            os.environ,
            {
                "DATABASE_CONNECT_ARGS": '{"timeout": 60, "sslmode": "require", "retries": 3, "keepalive": true}'
            },
        ):
            config = RelationalConfig()
            assert config.database_connect_args == (
                ("keepalive", True),
                ("retries", 3),
                ("sslmode", "require"),
                ("timeout", 60),
            )


_RESOLVER_LOGGER = "cognee.infrastructure.databases.utils.resolve_postgres_connection.logger"


def test_unknown_db_env_var_warns(tmp_path):
    """A typo'd DB_* var in the .env file is reported (not silently ignored),
    and the config still constructs (extra="allow", not forbid)."""
    envf = tmp_path / ".env"
    envf.write_text("DB_HSOT=x\n")

    with patch(_RESOLVER_LOGGER) as mock_logger:
        config = RelationalConfig(_env_file=str(envf))

    assert config.db_provider == "sqlite"  # still constructs with defaults
    warned = [call.args[1] for call in mock_logger.warning.call_args_list]
    assert warned == ["db_hsot"]


def test_relational_does_not_false_warn_on_other_namespaces(tmp_path):
    """The DB_ scan must not warn on VECTOR_DB_*/GRAPH_DATABASE_*/unrelated keys
    that land in relational's model_extra (locks the prefix-collision fix)."""
    envf = tmp_path / ".env"
    envf.write_text("VECTOR_DB_HOST=v\nGRAPH_DATABASE_FOO=g\nLLM_API_KEY=k\n")

    with patch(_RESOLVER_LOGGER) as mock_logger:
        RelationalConfig(_env_file=str(envf))

    mock_logger.warning.assert_not_called()
