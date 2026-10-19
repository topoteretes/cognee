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
async def test_sources_are_retrieved_concurrently_and_a_slow_one_degrades_only_itself(
    user, monkeypatch
):
    import asyncio

    fast_a, fast_b, slow = sql_target(), sql_target(), sql_target()
    fast_a["connection"], fast_b["connection"], slow["connection"] = "a", "b", "slow"
    monkeypatch.setattr(
        search, "route_sources", AsyncMock(return_value={"targets": [fast_a, slow, fast_b]})
    )
    monkeypatch.setattr(search, "PER_SOURCE_TIMEOUT_SECONDS", 0.3)
    in_flight, peak = 0, 0

    async def sql(user_id, connection, question):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            # The slow source would run far past the overall budget if awaited alone.
            await asyncio.sleep(600 if connection == "slow" else 0.05)
        finally:
            in_flight -= 1
        return TextToSqlResult(
            connection=connection,
            dialect="db",
            question=question,
            success=True,
            sql="SELECT 1",
            rows=[{"from": connection}],
            row_count=1,
        )

    monkeypatch.setattr(search, "run_text_to_sql", sql)
    result = await search.search_sources(user, "count")
    # All three sources were in flight at once, and the slow one was cut at its
    # own bound; evidence keeps routing order regardless of completion order.
    assert peak == 3
    assert [hit["structured"]["rows"][0]["from"] for hit in result["evidence"]] == ["a", "b"]
    assert result["errors"] == [
        {
            "source_id": slow["id"],
            "retrieval_method": "sql",
            "error": "Source retrieval failed or timed out.",
        }
    ]
    assert [t["id"] for t in result["coverage"]["searched_targets"]] == [fast_a["id"], fast_b["id"]]


@pytest.mark.asyncio
async def test_node_set_mismatch_is_rejected_without_failing_the_search(user, monkeypatch):
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
        search,
        "source_documents_by_id",
        AsyncMock(return_value={str(document): {"node_sets": ["unselected"]}}),
    )
    result = await search.search_sources(user, "q")
    # The dataset was authorized; the stale node-set tag drops the hit, not the search.
    assert result["evidence"] == [] and result["errors"] == []
    assert result["coverage"]["rejected_unverifiable_hits"] == 1
    assert [t["id"] for t in result["coverage"]["searched_targets"]] == [target["id"]]


@pytest.mark.asyncio
async def test_document_metadata_is_loaded_once_per_source(user, monkeypatch):
    dataset, first, second, orphan = uuid4(), uuid4(), uuid4(), uuid4()
    target = {
        "id": str(uuid4()),
        "name": "notes",
        "dataset_id": str(dataset),
        "node_sets": [],
        "retrieval_method": "chunks",
    }
    monkeypatch.setattr(search, "route_sources", AsyncMock(return_value={"targets": [target]}))
    hits = [
        {"payload": {"document_id": str(first), "text": "one"}, "score": 0.9},
        {"payload": {"document_id": str(second), "text": "two"}, "score": 0.8},
        {"payload": {"document_id": str(first), "text": "three"}, "score": 0.7},
        {"payload": {"document_id": str(orphan), "text": "orphan chunk"}, "score": 0.6},
    ]
    monkeypatch.setattr(search, "native_search", AsyncMock(return_value=[{"objects_result": hits}]))
    rows = {
        str(first): {"id": str(first), "dataset_id": str(dataset), "node_sets": []},
        str(second): {"id": str(second), "dataset_id": str(dataset), "node_sets": []},
    }
    load = AsyncMock(return_value=rows)
    monkeypatch.setattr(search, "source_documents_by_id", load)
    result = await search.search_sources(user, "q")
    # One authorization + one row load for the whole source, not one per hit.
    load.assert_awaited_once()
    assert load.await_args.args[1] == dataset
    assert set(load.await_args.args[2]) == {str(first), str(second), str(orphan)}
    assert [hit["text"] for hit in result["evidence"]] == ["one", "two", "three"]
    # A chunk whose document no longer exists is unverifiable, not a 403.
    assert result["coverage"]["rejected_unverifiable_hits"] == 1


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
