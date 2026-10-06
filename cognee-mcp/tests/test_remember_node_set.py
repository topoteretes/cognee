"""node_set rides remember() through the MCP tool to every backend path (SDK-336).

The server tool takes ``node_set``; the client forwards it as the typed entry's
field in API session mode, as form fields in API permanent mode, and as the
call-level kwarg in local mode. Session writes pin the session's node set on
the server, so the plugins can keep per-project memory through MCP as well.
"""

import asyncio
import json
import sys
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastmcp import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import server
from src.cognee_client import CogneeClient


@pytest.mark.asyncio
async def test_api_session_mode_puts_node_set_on_the_typed_entry():
    seen = {}

    async def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(await request.aread())
        return httpx.Response(200, json={"status": "session_stored"})

    client = CogneeClient(api_url="http://localhost:8000", api_token="test-token")
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client.client:
        await client.remember("hello", session_id="s", node_set=["project-a"])

    assert seen["path"] == "/api/v1/remember/entry"
    assert seen["body"]["entry"]["type"] == "qa"
    assert seen["body"]["entry"]["node_set"] == ["project-a"]


@pytest.mark.asyncio
async def test_api_session_mode_omits_the_field_without_a_node_set():
    seen = {}

    async def handler(request):
        seen["body"] = json.loads(await request.aread())
        return httpx.Response(200, json={"status": "session_stored"})

    client = CogneeClient(api_url="http://localhost:8000", api_token="test-token")
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client.client:
        await client.remember("hello", session_id="s")

    assert "node_set" not in seen["body"]["entry"]


@pytest.mark.asyncio
async def test_api_session_conflict_surfaces_as_an_http_error():
    """A 409 from the server is not swallowed: the tool call fails loudly."""

    async def handler(request):
        return httpx.Response(
            409, json={"detail": {"message": "cannot change [SessionNodeSetConflictError]"}}
        )

    client = CogneeClient(api_url="http://localhost:8000", api_token="test-token")
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client.client:
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await client.remember("hello", session_id="s", node_set=["project-b"])

    assert raised.value.response.status_code == 409


@pytest.mark.asyncio
async def test_api_permanent_mode_sends_node_set_as_form_fields():
    seen = {}

    async def handler(request):
        message = BytesParser(policy=default).parsebytes(
            b"Content-Type: "
            + request.headers["content-type"].encode()
            + b"\r\n\r\n"
            + await request.aread()
        )
        seen["node_set"] = [
            p.get_payload(decode=True).decode()
            for p in message.iter_parts()
            if p.get_param("name", header="content-disposition") == "node_set"
        ]
        seen["path"] = request.url.path
        return httpx.Response(200, json={"status": "completed"})

    client = CogneeClient(api_url="http://localhost:8000", api_token="test-token")
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client.client:
        await client.remember("hello", node_set=["project-a", "project-b"])

    assert seen["path"] == "/api/v1/remember"
    assert seen["node_set"] == ["project-a", "project-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", [None, "s"])
async def test_local_mode_passes_the_call_level_kwarg(session_id):
    client = CogneeClient.__new__(CogneeClient)
    client.use_api = False
    client.cognee = SimpleNamespace(remember=AsyncMock())

    await client.remember("hello", session_id=session_id, node_set=["project-a"])

    kwargs = client.cognee.remember.call_args.kwargs
    assert kwargs["node_set"] == ["project-a"]
    assert kwargs.get("session_id") == session_id


@pytest.mark.asyncio
async def test_local_mode_omits_the_kwarg_without_a_node_set():
    client = CogneeClient.__new__(CogneeClient)
    client.use_api = False
    client.cognee = SimpleNamespace(remember=AsyncMock())

    await client.remember("hello")

    client.cognee.remember.assert_awaited_once_with(data="hello", dataset_name="main_dataset")


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
async def test_mcp_tool_forwards_node_set(monkeypatch, background):
    remember = AsyncMock(return_value={"status": "completed"})
    monkeypatch.setattr(server, "cognee_client", SimpleNamespace(remember=remember))
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "remember")
        assert "node_set" in tool.inputSchema["properties"]
        result = await client.call_tool(
            "remember", {"data": "hello", "node_set": ["project-a"], "background": background}
        )
        assert not result.is_error
        await asyncio.sleep(0)
    remember.assert_awaited_once()
    assert remember.call_args.kwargs["node_set"] == ["project-a"]


@pytest.mark.asyncio
async def test_mcp_tool_forwards_node_set_with_a_session(monkeypatch):
    remember = AsyncMock(return_value={"status": "session_stored"})
    monkeypatch.setattr(server, "cognee_client", SimpleNamespace(remember=remember))
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "remember", {"data": "hello", "session_id": "s", "node_set": ["project-a"]}
        )
        assert not result.is_error
    kwargs = remember.call_args.kwargs
    assert kwargs["session_id"] == "s"
    assert kwargs["node_set"] == ["project-a"]
