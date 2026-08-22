import pytest

from cognee.modules.tools.config import ToolsConfig


def test_tool_calls_disabled_by_default():
    config = ToolsConfig(_env_file=None)
    assert config.tool_calls_enabled is False


def test_default_limits():
    config = ToolsConfig(_env_file=None)
    assert config.text_to_sql_max_rows == 100
    assert config.text_to_sql_max_attempts == 3
    assert config.text_to_sql_statement_timeout_ms == 5000
    assert config.text_to_sql_max_schema_tables == 50


def test_sql_connections_empty_by_default():
    assert ToolsConfig(_env_file=None).sql_connections() == {}


def test_sql_connections_string_shorthand():
    config = ToolsConfig(_env_file=None, tool_sql_connections='{"a": "sqlite:///x.db"}')
    assert config.sql_connections() == {"a": {"connection_string": "sqlite:///x.db"}}


def test_sql_connections_object_form():
    config = ToolsConfig(
        _env_file=None,
        tool_sql_connections=(
            '{"analytics": {"connection_string": "postgresql://u:p@h/db",'
            ' "allowed_tables": ["orders"], "max_rows": 7}}'
        ),
    )
    connections = config.sql_connections()
    assert connections["analytics"]["max_rows"] == 7
    assert connections["analytics"]["allowed_tables"] == ["orders"]


def test_sql_connections_rejects_non_object():
    config = ToolsConfig(_env_file=None, tool_sql_connections='["not", "a", "dict"]')
    with pytest.raises(ValueError):
        config.sql_connections()


def test_sql_connections_rejects_entry_without_connection_string():
    config = ToolsConfig(_env_file=None, tool_sql_connections='{"a": {"max_rows": 5}}')
    with pytest.raises(ValueError):
        config.sql_connections()


def test_env_gate_parses(monkeypatch):
    monkeypatch.setenv("TOOL_CALLS_ENABLED", "true")
    assert ToolsConfig(_env_file=None).tool_calls_enabled is True
