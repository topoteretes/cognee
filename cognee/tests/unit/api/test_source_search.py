from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.data import source_catalog as catalog
from cognee.modules.data import source_search as search
from cognee.modules.tools.errors import ToolConnectionNotFoundError
from cognee.modules.tools.text_to_sql.engine import TextToSqlResult


@pytest.fixture
def user():
    return SimpleNamespace(id=uuid4())


def sql_target():
    return {
        "id": str(uuid4()),
        "name": "arbitrary analytics",
        "connection": "finance_ro",
        "retrieval_method": "sql",
        "dataset_id": None,
        "node_sets": [],
    }


@pytest.mark.asyncio
async def test_database_capability_executes_native_sql_not_chunks(user, monkeypatch):
    target = sql_target()
    monkeypatch.setattr(search, "route_sources", AsyncMock(return_value={"targets": [target]}))
    sql = AsyncMock(
        return_value=TextToSqlResult(
            connection="finance_ro",
            dialect="unknown",
            question="count",
            success=True,
            sql="SELECT COUNT(*) AS count FROM orders",
            rows=[{"count": 42}],
            row_count=1,
        )
    )
    monkeypatch.setattr(search, "run_text_to_sql", sql)
    chunks = AsyncMock(side_effect=AssertionError("SQL must not use CHUNKS"))
    monkeypatch.setattr(search, "native_search", chunks)
    result = await search.search_sources(user, "count orders")
    sql.assert_awaited_once_with(user.id, "finance_ro", "count orders")
    chunks.assert_not_awaited()
    assert result["evidence"][0]["structured"]["rows"] == [{"count": 42}]
    assert result["evidence"][0]["retrieval_method"] == "sql"
    assert result["coverage"]["complete"] is False


@pytest.mark.asyncio
async def test_revoked_connection_fails_closed(user, monkeypatch):
    monkeypatch.setattr(
        search, "route_sources", AsyncMock(return_value={"targets": [sql_target()]})
    )
    sql = AsyncMock(side_effect=ToolConnectionNotFoundError("private connection"))
    monkeypatch.setattr(search, "run_text_to_sql", sql)
    with pytest.raises(PermissionError):
        await search.search_sources(user, "count")
    sql.assert_awaited_once()


@pytest.mark.asyncio
async def test_sql_failure_is_not_an_empty_success_and_does_not_leak(user, monkeypatch):
    monkeypatch.setattr(
        search, "route_sources", AsyncMock(return_value={"targets": [sql_target()]})
    )
    monkeypatch.setattr(
        search,
        "run_text_to_sql",
        AsyncMock(
            return_value=TextToSqlResult(
                connection="finance",
                dialect="db",
                question="count",
                success=False,
                error="secret DSN",
            )
        ),
    )
    result = await search.search_sources(user, "count")
    assert result["errors"] and not result["evidence"]
    assert not result["coverage"]["searched_targets"]
    assert "secret DSN" not in str(result)


@pytest.mark.asyncio
async def test_connection_discovery_projects_only_safe_metadata(user, monkeypatch):
    from cognee.modules.tools import config, connections

    monkeypatch.setattr(
        config, "get_tools_config", lambda: SimpleNamespace(tool_calls_enabled=True)
    )
    method = AsyncMock(
        return_value=[
            {
                "name": "my DB",
                "provider": "never-seen-provider",
                "description": "orders",
                "allowed_tables": ["orders"],
                "connection_string": "secret",
                "ciphertext": "secret",
            }
        ]
    )
    monkeypatch.setattr(connections, "list_tool_connections", method)
    result = await catalog.connection_descriptors(user)
    method.assert_awaited_once_with(user.id)
    assert result[0]["retrieval_method"] == "sql"
    assert result[0]["source_name"] == "never-seen-provider"
    assert "secret" not in str(result)


@pytest.mark.asyncio
async def test_dataset_restriction_does_not_expand_to_connections(user, monkeypatch):
    monkeypatch.setattr(catalog, "get_all_user_permission_datasets", AsyncMock(return_value=[]))
    connections = AsyncMock(side_effect=AssertionError("must not widen"))
    monkeypatch.setattr(catalog, "connection_descriptors", connections)
    assert await catalog.source_catalog(user, [], include_connections=True) == {
        "items": [],
        "complete": True,
    }
    connections.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_tool_gate_never_lists_connections(user, monkeypatch):
    from cognee.modules.tools import config, connections

    monkeypatch.setattr(
        config, "get_tools_config", lambda: SimpleNamespace(tool_calls_enabled=False)
    )
    method = AsyncMock(side_effect=AssertionError("disabled"))
    monkeypatch.setattr(connections, "list_tool_connections", method)
    assert await catalog.connection_descriptors(user) == []
    method.assert_not_awaited()


@pytest.mark.asyncio
async def test_unverifiable_document_membership_fails_closed(user, monkeypatch):
    dataset, document = uuid4(), uuid4()
    target = {
        "id": str(uuid4()),
        "name": "private notes",
        "dataset_id": str(dataset),
        "node_sets": ["selected"],
        "retrieval_method": "chunks",
    }
    monkeypatch.setattr(search, "route_sources", AsyncMock(return_value={"targets": [target]}))
    monkeypatch.setattr(
        search,
        "native_search",
        AsyncMock(
            return_value=[
                {
                    "objects_result": [
                        {"payload": {"document_id": str(document), "text": "wrong scope"}}
                    ]
                }
            ]
        ),
    )
    monkeypatch.setattr(
        search, "source_document", AsyncMock(return_value={"node_sets": ["unselected"]})
    )
    with pytest.raises(PermissionError):
        await search.search_sources(user, "q")


@pytest.mark.asyncio
async def test_served_sdk_never_falls_back_to_embedded(monkeypatch):
    from cognee.api.v1 import sources
    from cognee.api.v1.serve import state

    remote = SimpleNamespace(search_sources=AsyncMock(side_effect=PermissionError("denied")))
    monkeypatch.setattr(state, "get_remote_client", lambda: remote)
    with pytest.raises(PermissionError):
        await sources.search("question", source_hint="unknown provider")
    remote.search_sources.assert_awaited_once()
    assert remote.search_sources.await_args.args[0]["source_hint"] == "unknown provider"
