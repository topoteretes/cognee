from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from src import server


@pytest.fixture(autouse=True)
def restore_tool_mode():
    saved = list(server.mcp._transforms)
    yield
    server.mcp._transforms = saved


@pytest.mark.asyncio
async def test_code_search_forwards_a_code_scoped_structured_query(monkeypatch):
    fake = SimpleNamespace(
        recall=AsyncMock(return_value=[{"source": "code", "facts": []}]),
        get_recall_state=AsyncMock(),
    )
    monkeypatch.setattr(server, "cognee_client", fake)

    await server.code_search(
        operation="query_facts",
        arguments={"kinds": ["module", "symbol"], "limit": 100},
        datasets="test-code-dataset",
    )

    assert fake.recall.call_args.kwargs == {
        "query_text": "",
        "search_type": "CODE",
        "datasets": ["test-code-dataset"],
        "session_id": None,
        "system_prompt": None,
        "top_k": 15,
        "scope": ["code"],
        "code_query": {
            "operation": "query_facts",
            "kinds": ["module", "symbol"],
            "limit": 100,
        },
    }


@pytest.mark.asyncio
async def test_code_search_schema_is_available_in_default_mode():
    server.apply_tool_mode("default")
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    schema = tools["code_search"].inputSchema
    properties = schema["properties"]
    assert {"operation", "arguments", "datasets", "query", "top_k"} <= set(properties)
    assert set(properties["operation"]["enum"]) == {
        "query_facts",
        "explore",
        "traverse",
        "find_path",
        "impact_analysis",
        "insights",
        "architecture",
        "delta",
    }


@pytest.mark.asyncio
async def test_code_search_rejects_operation_inside_arguments(monkeypatch):
    fake = SimpleNamespace(recall=AsyncMock(), get_recall_state=AsyncMock())
    monkeypatch.setattr(server, "cognee_client", fake)

    result = await server.code_search(
        operation="query_facts",
        arguments={"operation": "explore"},
    )

    assert result[0].text.startswith("Error:")
    assert fake.recall.await_count == 0
