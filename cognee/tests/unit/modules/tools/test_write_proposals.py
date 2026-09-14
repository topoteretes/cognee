import sqlite3
from uuid import uuid4

import pytest

import cognee.modules.tools.text_to_sql.write_proposals as wp_mod
from cognee.modules.tools.config import ToolsConfig
from cognee.modules.tools.errors import (
    ToolCallsDisabledError,
    ToolError,
    ToolWriteNotAllowedError,
)
from cognee.modules.tools.text_to_sql.write_proposals import (
    apply_write_proposal,
    list_write_proposals,
    propose_corrections_from_contradictions,
    propose_sql_write,
    reject_write_proposal,
)


def _read_rows(db_path, query):
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


@pytest.fixture
def source_db(tmp_path):
    db_path = tmp_path / "source.db"
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
def relational_engine(monkeypatch, tmp_path):
    """Point the proposal store (and connections module) at a temp cognee DB."""
    import cognee.modules.tools.connections as connections_mod
    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )

    adapter = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path}/cognee.db")
    monkeypatch.setattr(connections_mod, "get_relational_engine", lambda: adapter)
    monkeypatch.setattr(connections_mod, "_table_ready", False)

    import cognee.infrastructure.databases.relational as relational_pkg

    monkeypatch.setattr(relational_pkg, "get_relational_engine", lambda: adapter)
    return adapter


@pytest.fixture
def write_enabled(monkeypatch):
    config = ToolsConfig(_env_file=None, tool_calls_enabled=True, tool_write_calls_enabled=True)
    monkeypatch.setattr(wp_mod, "get_tools_config", lambda: config)
    return config


@pytest.fixture
def writable_connection(monkeypatch, source_db):
    async def _get_tool_connection(user_id, name):
        return {
            "name": name,
            "provider": "sqlite",
            "connection_string": f"sqlite:///{source_db}",
            "allowed_tables": None,
            "max_rows": None,
            "allow_writes": True,
        }

    monkeypatch.setattr(wp_mod, "get_tool_connection", _get_tool_connection)


def _mock_llm(monkeypatch, responses):
    calls = {"count": 0}

    async def _acreate(text_input, system_prompt, response_model):
        response = responses[min(calls["count"], len(responses) - 1)]
        calls["count"] += 1
        return response

    monkeypatch.setattr(wp_mod.LLMGateway, "acreate_structured_output", staticmethod(_acreate))
    return calls


@pytest.mark.asyncio
async def test_gates_off_raises(monkeypatch, writable_connection):
    config = ToolsConfig(_env_file=None, tool_calls_enabled=True, tool_write_calls_enabled=False)
    monkeypatch.setattr(wp_mod, "get_tools_config", lambda: config)
    with pytest.raises(ToolCallsDisabledError):
        await propose_sql_write(uuid4(), "source", "fix it")


@pytest.mark.asyncio
async def test_connection_without_allow_writes_raises(monkeypatch, write_enabled, source_db):
    async def _get_tool_connection(user_id, name):
        return {
            "name": name,
            "provider": "sqlite",
            "connection_string": f"sqlite:///{source_db}",
            "allow_writes": False,
        }

    monkeypatch.setattr(wp_mod, "get_tool_connection", _get_tool_connection)
    with pytest.raises(ToolWriteNotAllowedError):
        await propose_sql_write(uuid4(), "source", "fix it")


@pytest.mark.asyncio
async def test_propose_does_not_modify_source(
    monkeypatch, write_enabled, writable_connection, relational_engine, source_db
):
    _mock_llm(monkeypatch, ["UPDATE customers SET country = 'fr' WHERE name = 'acme'"])

    proposal = await propose_sql_write(uuid4(), "source", "acme is in France, not Germany")

    assert proposal["status"] == "proposed"
    assert proposal["estimated_rows"] == 1
    assert proposal["target_table"] == "customers"
    # Dry-run rolled back: source data unchanged.
    assert _read_rows(source_db, "SELECT country FROM customers WHERE name='acme'") == [("de",)]


@pytest.mark.asyncio
async def test_apply_commits_and_records(
    monkeypatch, write_enabled, writable_connection, relational_engine, source_db
):
    _mock_llm(monkeypatch, ["UPDATE customers SET country = 'fr' WHERE name = 'acme'"])
    user_id = uuid4()

    proposal = await propose_sql_write(user_id, "source", "acme is in France")
    applied = await apply_write_proposal(user_id, proposal["proposal_id"])

    assert applied["status"] == "applied"
    assert applied["applied_rows"] == 1
    assert _read_rows(source_db, "SELECT country FROM customers WHERE name='acme'") == [("fr",)]

    # Double-apply is refused.
    with pytest.raises(ToolError, match="applied"):
        await apply_write_proposal(user_id, proposal["proposal_id"])


@pytest.mark.asyncio
async def test_apply_by_non_owner_fails_closed(
    monkeypatch, write_enabled, writable_connection, relational_engine, source_db
):
    from cognee.modules.tools.errors import ToolConnectionNotFoundError

    _mock_llm(monkeypatch, ["UPDATE customers SET country = 'fr' WHERE name = 'acme'"])
    proposal = await propose_sql_write(uuid4(), "source", "acme is in France")

    with pytest.raises(ToolConnectionNotFoundError):
        await apply_write_proposal(uuid4(), proposal["proposal_id"])
    assert _read_rows(source_db, "SELECT country FROM customers WHERE name='acme'") == [("de",)]


@pytest.mark.asyncio
async def test_reject_blocks_apply(
    monkeypatch, write_enabled, writable_connection, relational_engine, source_db
):
    _mock_llm(monkeypatch, ["UPDATE customers SET country = 'fr' WHERE name = 'acme'"])
    user_id = uuid4()

    proposal = await propose_sql_write(user_id, "source", "acme is in France")
    rejected = await reject_write_proposal(user_id, proposal["proposal_id"])
    assert rejected["status"] == "rejected"

    with pytest.raises(ToolError, match="rejected"):
        await apply_write_proposal(user_id, proposal["proposal_id"])
    assert _read_rows(source_db, "SELECT country FROM customers WHERE name='acme'") == [("de",)]


@pytest.mark.asyncio
async def test_affected_rows_cap_rolls_back(
    monkeypatch, write_enabled, writable_connection, relational_engine, source_db
):
    config = ToolsConfig(
        _env_file=None,
        tool_calls_enabled=True,
        tool_write_calls_enabled=True,
        text_to_sql_max_affected_rows=1,
    )
    monkeypatch.setattr(wp_mod, "get_tools_config", lambda: config)
    # Touches both rows — over the cap of 1.
    _mock_llm(monkeypatch, ["UPDATE customers SET country = 'xx' WHERE id > 0"])
    user_id = uuid4()

    proposal = await propose_sql_write(user_id, "source", "wipe countries")
    result = await apply_write_proposal(user_id, proposal["proposal_id"])

    assert result["status"] == "failed"
    assert "cap" in result["error"]
    assert _read_rows(source_db, "SELECT DISTINCT country FROM customers ORDER BY country") == [
        ("de",),
        ("us",),
    ]


@pytest.mark.asyncio
async def test_estimate_mismatch_rolls_back(
    monkeypatch, write_enabled, writable_connection, relational_engine, source_db
):
    _mock_llm(monkeypatch, ["UPDATE customers SET country = 'fr' WHERE country = 'de'"])
    user_id = uuid4()
    proposal = await propose_sql_write(user_id, "source", "fix German customers")
    assert proposal["estimated_rows"] == 1

    # Data changes between review and apply: a second German customer appears.
    connection = sqlite3.connect(source_db)
    connection.execute("INSERT INTO customers (name, country) VALUES ('newco', 'de')")
    connection.commit()
    connection.close()

    result = await apply_write_proposal(user_id, proposal["proposal_id"])

    assert result["status"] == "failed"
    assert "estimated" in result["error"]
    assert _read_rows(source_db, "SELECT count(*) FROM customers WHERE country='fr'") == [(0,)]


@pytest.mark.asyncio
async def test_not_applicable_raises_cleanly(
    monkeypatch, write_enabled, writable_connection, relational_engine
):
    _mock_llm(monkeypatch, ["NOT_APPLICABLE"])
    with pytest.raises(ToolError, match="cannot be expressed"):
        await propose_sql_write(uuid4(), "source", "fix the weather")


@pytest.mark.asyncio
async def test_guard_rejection_feeds_back_and_retries(
    monkeypatch, write_enabled, writable_connection, relational_engine, source_db
):
    calls = _mock_llm(
        monkeypatch,
        [
            "DELETE FROM customers WHERE id = 1",
            "UPDATE customers SET country = 'fr' WHERE id = 1",
        ],
    )

    proposal = await propose_sql_write(uuid4(), "source", "fix customer 1")

    assert proposal["status"] == "proposed"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_propose_corrections_from_contradictions(
    monkeypatch, write_enabled, writable_connection, relational_engine, source_db
):
    _mock_llm(monkeypatch, ["UPDATE customers SET country = 'fr' WHERE name = 'acme'"])

    class FakeGraphEngine:
        async def get_graph_data(self):
            edges = [
                ("n1", "n2", "is_part_of", {}),
                (
                    "n3",
                    "n4",
                    "contradicts",
                    {
                        "first_fact": "acme located in de",
                        "second_fact": "acme located in fr",
                        "reason": "a company HQ is in one country",
                        "confidence": 0.9,
                    },
                ),
            ]
            return [], edges

    async def fake_get_graph_engine():
        return FakeGraphEngine()

    import cognee.infrastructure.databases.graph as graph_pkg

    monkeypatch.setattr(graph_pkg, "get_graph_engine", fake_get_graph_engine)

    user_id = uuid4()
    proposals = await propose_corrections_from_contradictions(user_id, "source")

    assert len(proposals) == 1
    assert proposals[0]["evidence"]["reason"] == "a company HQ is in one country"
    assert proposals[0]["status"] == "proposed"

    listed = await list_write_proposals(user_id, status="proposed")
    assert len(listed) == 1
