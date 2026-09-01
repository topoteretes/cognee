"""Wiring tests for the recall "tools" scope (text-to-SQL tool calls)."""

import importlib
import types
from uuid import uuid4

import pytest
from pydantic import TypeAdapter

from cognee.exceptions import CogneeValidationError
from cognee.memory.entries import normalize_scope
from cognee.modules.recall.types.RecallResponse import RecallResponse, ResponseToolEntry
from cognee.modules.tools.config import ToolsConfig


def _make_user(user_id=None, tenant_id=None):
    return types.SimpleNamespace(id=user_id or uuid4(), tenant_id=tenant_id)


@pytest.fixture
def api_recall_mod():
    return importlib.import_module("cognee.api.v1.recall.recall")


@pytest.fixture
def no_remote_client(monkeypatch):
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)


@pytest.fixture
def tools_enabled(monkeypatch):
    config_mod = importlib.import_module("cognee.modules.tools.config")
    monkeypatch.setattr(
        config_mod, "get_tools_config", lambda: ToolsConfig(_env_file=None, tool_calls_enabled=True)
    )


class TestScopeNormalization:
    def test_tools_is_a_valid_scope(self):
        assert normalize_scope(["tools"]) == ["tools"]
        assert normalize_scope(["graph", "tools"]) == ["graph", "tools"]

    def test_all_does_not_expand_to_tools(self):
        assert "tools" not in normalize_scope("all")

    def test_all_preserves_explicit_tools(self):
        assert "tools" in normalize_scope(["all", "tools"])

    def test_unknown_scope_error_lists_tools(self):
        with pytest.raises(ValueError, match="tools"):
            normalize_scope("bogus")


class TestResponseUnion:
    def test_tool_entry_round_trips_through_union(self):
        adapter = TypeAdapter(list[RecallResponse])
        entries = adapter.validate_python(
            [
                {
                    "source": "tools",
                    "tool_name": "text_to_sql",
                    "question": "q",
                    "text": "42",
                    "success": True,
                    "structured": {"sql": "SELECT 42", "rows": [{"n": 42}]},
                }
            ]
        )
        assert isinstance(entries[0], ResponseToolEntry)
        assert entries[0].structured["rows"] == [{"n": 42}]

    def test_unknown_source_rejected(self):
        adapter = TypeAdapter(list[RecallResponse])
        with pytest.raises(Exception):
            adapter.validate_python([{"source": "nonsense", "text": "x"}])


@pytest.mark.asyncio
async def test_tools_scope_with_gate_off_raises(monkeypatch, api_recall_mod, no_remote_client):
    config_mod = importlib.import_module("cognee.modules.tools.config")
    monkeypatch.setattr(
        config_mod,
        "get_tools_config",
        lambda: ToolsConfig(_env_file=None, tool_calls_enabled=False),
    )

    with pytest.raises(CogneeValidationError, match="TOOL_CALLS_ENABLED"):
        await api_recall_mod.recall(query_text="q", scope=["tools"], user=_make_user())


@pytest.mark.asyncio
async def test_tools_scope_runs_requested_connections(
    monkeypatch, api_recall_mod, no_remote_client, tools_enabled
):
    text_to_sql_mod = importlib.import_module("cognee.modules.tools.text_to_sql")
    engine_mod = importlib.import_module("cognee.modules.tools.text_to_sql.engine")

    calls = []

    async def fake_run_text_to_sql(user_id, connection_name, question):
        calls.append(connection_name)
        success = connection_name != "dead_db"
        return engine_mod.TextToSqlResult(
            connection=connection_name,
            dialect="sqlite",
            question=question,
            success=success,
            sql="SELECT 1" if success else None,
            rows=[{"n": 1}] if success else [],
            row_count=1 if success else 0,
            attempts=1,
            error=None if success else "connection refused",
        )

    monkeypatch.setattr(text_to_sql_mod, "run_text_to_sql", fake_run_text_to_sql)

    results = await api_recall_mod.recall(
        query_text="how many?",
        scope=["tools"],
        tool_connections=["analytics", "dead_db"],
        user=_make_user(),
    )

    assert calls == ["analytics", "dead_db"]
    assert len(results) == 2
    assert all(entry.source == "tools" for entry in results)
    ok, dead = results
    assert ok.success is True and ok.structured["rows"] == [{"n": 1}]
    # A failing connection yields an error entry instead of aborting recall.
    assert dead.success is False and "refused" in dead.error


@pytest.mark.asyncio
async def test_tools_scope_defaults_to_all_visible_connections(
    monkeypatch, api_recall_mod, no_remote_client, tools_enabled
):
    connections_mod = importlib.import_module("cognee.modules.tools.connections")
    text_to_sql_mod = importlib.import_module("cognee.modules.tools.text_to_sql")
    engine_mod = importlib.import_module("cognee.modules.tools.text_to_sql.engine")

    async def fake_list(user_id):
        return [{"name": "a"}, {"name": "b"}]

    async def fake_run(user_id, connection_name, question):
        return engine_mod.TextToSqlResult(
            connection=connection_name,
            dialect="sqlite",
            question=question,
            success=True,
            sql="SELECT 1",
            rows=[],
            attempts=1,
        )

    monkeypatch.setattr(connections_mod, "list_tool_connections", fake_list)
    monkeypatch.setattr(text_to_sql_mod, "run_text_to_sql", fake_run)

    results = await api_recall_mod.recall(query_text="q", scope=["tools"], user=_make_user())

    assert {entry.structured["connection"] for entry in results} == {"a", "b"}


@pytest.mark.asyncio
async def test_tools_scope_with_no_connections_returns_empty(
    monkeypatch, api_recall_mod, no_remote_client, tools_enabled
):
    connections_mod = importlib.import_module("cognee.modules.tools.connections")

    async def fake_list(user_id):
        return []

    monkeypatch.setattr(connections_mod, "list_tool_connections", fake_list)

    results = await api_recall_mod.recall(query_text="q", scope=["tools"], user=_make_user())
    assert results == []


@pytest.mark.asyncio
async def test_tools_trigger_on_empty_skips_tools_when_other_sources_hit(
    monkeypatch, api_recall_mod, no_remote_client, tools_enabled
):
    """'on_empty' = go back to the source DB only when cognee lacks context."""
    text_to_sql_mod = importlib.import_module("cognee.modules.tools.text_to_sql")
    engine_mod = importlib.import_module("cognee.modules.tools.text_to_sql.engine")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")

    tool_calls = []

    async def fake_run(user_id, connection_name, question):
        tool_calls.append(connection_name)
        return engine_mod.TextToSqlResult(
            connection=connection_name,
            dialect="sqlite",
            question=question,
            success=True,
            sql="SELECT 1",
            rows=[{"n": 1}],
            row_count=1,
            attempts=1,
        )

    monkeypatch.setattr(text_to_sql_mod, "run_text_to_sql", fake_run)

    from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
    from cognee.modules.search.types import SearchType

    graph_payload = SearchResultPayload(
        search_type=SearchType.GRAPH_COMPLETION, completion="a graph answer"
    )

    async def graph_with_results(**kwargs):
        return [graph_payload]

    async def graph_empty(**kwargs):
        return []

    async def dummy_set_user(_user):
        return None

    monkeypatch.setattr(api_recall_mod, "set_session_user_context_variable", dummy_set_user)

    # Graph has context → tools (listed first, even) must not run.
    monkeypatch.setattr(search_methods, "authorized_search", graph_with_results)
    results = await api_recall_mod.recall(
        query_text="q",
        scope=["tools", "graph"],
        tool_connections=["analytics"],
        tools_trigger="on_empty",
        auto_route=False,
        user=_make_user(),
    )
    assert tool_calls == []
    assert all(entry.source == "graph" for entry in results)

    # Graph lacks context → tools run as the fallback.
    monkeypatch.setattr(search_methods, "authorized_search", graph_empty)
    results = await api_recall_mod.recall(
        query_text="q",
        scope=["graph", "tools"],
        tool_connections=["analytics"],
        tools_trigger="on_empty",
        auto_route=False,
        user=_make_user(),
    )
    assert tool_calls == ["analytics"]
    assert [entry.source for entry in results] == ["tools"]


@pytest.mark.asyncio
async def test_invalid_tools_trigger_rejected(api_recall_mod):
    with pytest.raises(CogneeValidationError, match="tools_trigger"):
        await api_recall_mod.recall(query_text="q", scope=["tools"], tools_trigger="bogus")


@pytest.mark.asyncio
async def test_invalid_context_format_rejected(api_recall_mod):
    with pytest.raises(CogneeValidationError, match="context_format"):
        await api_recall_mod.recall(query_text="q", only_context=True, context_format="bogus")


@pytest.mark.asyncio
async def test_remote_client_forwards_tool_connections(monkeypatch, api_recall_mod):
    captured = {}

    async def dummy_remote_recall(query_text, query_type, **kwargs):
        captured.update(kwargs)
        return []

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(
        serve_state,
        "get_remote_client",
        lambda: types.SimpleNamespace(recall=dummy_remote_recall),
    )

    await api_recall_mod.recall(
        query_text="q",
        scope=["tools"],
        tool_connections=["analytics"],
        user=_make_user(),
    )

    assert captured["tool_connections"] == ["analytics"]
    assert captured["scope"] == ["tools"]
