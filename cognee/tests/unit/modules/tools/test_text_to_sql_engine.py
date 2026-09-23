import sqlite3

import pytest

import cognee.modules.tools.text_to_sql.engine as engine_mod
from cognee.modules.tools.config import ToolsConfig
from cognee.modules.tools.errors import ToolCallsDisabledError
from cognee.modules.tools.text_to_sql.engine import run_text_to_sql


@pytest.fixture
def sample_db(tmp_path):
    db_path = tmp_path / "warehouse.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        INSERT INTO customers (name, country) VALUES ('acme', 'de'), ('globex', 'us');
        """
    )
    connection.commit()
    connection.close()
    return db_path


@pytest.fixture
def enabled_config(monkeypatch):
    config = ToolsConfig(_env_file=None, tool_calls_enabled=True)
    monkeypatch.setattr(engine_mod, "get_tools_config", lambda: config)
    return config


@pytest.fixture
def fake_connection(monkeypatch, sample_db):
    async def _get_tool_connection(user_id, name):
        return {
            "name": name,
            "provider": "sqlite",
            "connection_string": f"sqlite:///{sample_db}",
            "allowed_tables": None,
            "max_rows": None,
        }

    monkeypatch.setattr(engine_mod, "get_tool_connection", _get_tool_connection)


def _mock_llm(monkeypatch, responses):
    """Feed canned SQL strings to the generation step, tracking call count."""
    calls = {"count": 0, "prompts": []}

    async def _generate(question, schema, dialect, max_rows, previous_attempts):
        calls["prompts"].append(previous_attempts)
        response = responses[min(calls["count"], len(responses) - 1)]
        calls["count"] += 1
        return response

    monkeypatch.setattr(engine_mod, "_generate_sql", _generate)
    return calls


@pytest.mark.asyncio
async def test_gate_off_raises(monkeypatch):
    config = ToolsConfig(_env_file=None, tool_calls_enabled=False)
    monkeypatch.setattr(engine_mod, "get_tools_config", lambda: config)
    with pytest.raises(ToolCallsDisabledError):
        await run_text_to_sql(None, "analytics", "how many customers?")


@pytest.mark.asyncio
async def test_happy_path(enabled_config, fake_connection, monkeypatch):
    calls = _mock_llm(
        monkeypatch, ["SELECT country, count(*) AS n FROM customers GROUP BY country"]
    )

    result = await run_text_to_sql(None, "warehouse", "customers per country?")

    assert result.success is True
    assert result.error is None
    assert result.row_count == 2
    assert result.attempts == 1
    assert "LIMIT" in result.sql
    assert calls["count"] == 1
    structured = result.structured()
    assert structured["connection"] == "warehouse"
    assert structured["rows"][0]["n"] == 1
    assert "connection_string" not in str(structured)


@pytest.mark.asyncio
async def test_empty_result_is_success_without_retry(enabled_config, fake_connection, monkeypatch):
    calls = _mock_llm(monkeypatch, ["SELECT * FROM customers WHERE country = 'jp'"])

    result = await run_text_to_sql(None, "warehouse", "japanese customers?")

    assert result.success is True
    assert result.rows == []
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_guard_rejection_feeds_back_and_retries(enabled_config, fake_connection, monkeypatch):
    calls = _mock_llm(
        monkeypatch,
        ["DELETE FROM customers", "SELECT name FROM customers"],
    )

    result = await run_text_to_sql(None, "warehouse", "list customers")

    assert result.success is True
    assert result.attempts == 2
    assert calls["count"] == 2
    # The second prompt must carry the first attempt's rejection.
    assert "Rejected" in calls["prompts"][1]


@pytest.mark.asyncio
async def test_execution_error_feeds_back_and_retries(enabled_config, fake_connection, monkeypatch):
    calls = _mock_llm(
        monkeypatch,
        ["SELECT nope FROM not_a_table", "SELECT name FROM customers"],
    )

    result = await run_text_to_sql(None, "warehouse", "list customers")

    assert result.success is True
    assert result.attempts == 2
    assert "error" in calls["prompts"][1].lower()


@pytest.mark.asyncio
async def test_exhausted_attempts_returns_error_result(
    enabled_config, fake_connection, monkeypatch
):
    _mock_llm(monkeypatch, ["DROP TABLE customers"])

    result = await run_text_to_sql(None, "warehouse", "destroy everything")

    assert result.success is False
    assert result.error is not None
    assert result.attempts == 3
    assert result.rows == []


def test_prompt_template_renders():
    from cognee.infrastructure.llm.prompts import render_prompt

    rendered = render_prompt(
        "text_to_sql_system.txt",
        context={
            "schema": {
                "orders": {
                    "columns": [{"name": "id", "type": "INTEGER"}],
                    "primary_key": ["id"],
                    "foreign_keys": [
                        {"column": "customer_id", "ref_table": "customers", "ref_column": "id"}
                    ],
                }
            },
            "dialect": "sqlite",
            "max_rows": 100,
            "previous_attempts": "No attempts yet",
        },
    )
    assert "orders" in rendered
    assert "sqlite" in rendered
    assert "100" in rendered
    assert "No attempts yet" in rendered
