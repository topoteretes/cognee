"""Recall forwards code-graph queries: scope and code_query reach the client."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from src import server


@pytest.mark.asyncio
async def test_recall_forwards_scope_and_code_query(monkeypatch):
    payload = [{"source": "code", "operation": "query_facts", "facts": [], "total": 0}]
    fake = SimpleNamespace(
        recall=AsyncMock(return_value=payload),
        get_recall_state=AsyncMock(),
    )
    monkeypatch.setattr(server, "cognee_client", fake)
    result = await server.recall(
        "list indexed language facts",
        datasets="test-code-dataset",
        scope="code",
        code_query={"operation": "query_facts", "property": "language", "limit": 500},
    )
    call = fake.recall.call_args.kwargs
    assert call["scope"] == ["code"]
    assert call["code_query"] == {"operation": "query_facts", "property": "language", "limit": 500}
    assert "[code]" in result[0].text


@pytest.mark.asyncio
async def test_recall_parses_scope_csv_and_defaults(monkeypatch):
    fake = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        get_recall_state=AsyncMock(),
    )
    monkeypatch.setattr(server, "cognee_client", fake)
    await server.recall("q", datasets="test-code-dataset", scope="code,session")
    call = fake.recall.call_args.kwargs
    assert call["scope"] == ["code", "session"]
    assert call["code_query"] is None
    await server.recall("q", datasets="test-code-dataset")
    call = fake.recall.call_args.kwargs
    assert call["scope"] is None


@pytest.mark.asyncio
async def test_recall_rejects_non_dict_code_query():
    result = await server.recall("q", code_query="query_facts")
    assert result[0].text.startswith("Error:")
    assert "code_query must be a JSON object" in result[0].text


@pytest.mark.asyncio
async def test_recall_tool_schema_exposes_scope_and_code_query():
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    schema = tools["recall"].inputSchema
    props = schema.get("properties", {})
    assert "scope" in props
    assert "code_query" in props
