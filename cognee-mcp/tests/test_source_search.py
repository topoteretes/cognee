import json
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from src import server
from src.cognee_client import CogneeClient


@pytest.mark.asyncio
async def test_mcp_source_tool_keeps_native_sql_evidence(monkeypatch):
    result = {
        "evidence": [
            {
                "retrieval_method": "sql",
                "structured": {"sql": "SELECT 1", "rows": [{"n": 1}]},
            }
        ]
    }
    client = SimpleNamespace(search_sources=AsyncMock(return_value=result))
    monkeypatch.setattr(server, "cognee_client", client)
    output = await server.search_sources("count", source_hint="an unfamiliar database")
    assert json.loads(output[0].text) == result
    client.search_sources.assert_awaited_once_with(
        "count", "an unfamiliar database", None, True, 10
    )


@pytest.mark.asyncio
async def test_mcp_api_denial_has_no_embedded_fallback():
    import httpx

    client = CogneeClient(api_url="http://localhost:8011", api_token="agent-key")
    seen = []

    async def handle(request):
        seen.append(request)
        return httpx.Response(403, json={"detail": "denied"})

    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.search_sources("count")
        assert len(seen) == 1
        assert seen[0].headers["Authorization"] == "Bearer agent-key"
        assert seen[0].url.path == "/api/v1/datasets/source-search"
    finally:
        await client.client.aclose()
