"""Per-write ontology selection across MCP, multipart HTTP, and local storage."""

import asyncio
import base64
import sys
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastmcp import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import server
from src.cognee_client import CogneeClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "keys,expected", [(None, []), ([], []), ("one", ["one"]), (["one", "two"], ["one", "two"])]
)
@pytest.mark.parametrize("upload", [False, True])
async def test_api_multipart(keys, expected, upload):
    async def handler(request):
        message = BytesParser(policy=default).parsebytes(
            b"Content-Type: "
            + request.headers["content-type"].encode()
            + b"\r\n\r\n"
            + await request.aread()
        )
        parts = list(message.iter_parts())
        values = [
            p.get_payload(decode=True).decode()
            for p in parts
            if p.get_param("name", header="content-disposition") == "ontology_key"
        ]
        assert values == expected
        assert request.url.path == "/api/v1/remember"
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(200, json={"status": "completed"})

    client = CogneeClient(api_url="http://localhost:8000", api_token="test-token")
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client.client:
        kwargs = (
            {
                "data": None,
                "filename": "note.txt",
                "content_base64": base64.b64encode(b"hello").decode(),
            }
            if upload
            else {"data": "hello"}
        )
        await client.remember(**kwargs, ontology_key=keys)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_api", [False, True])
async def test_session_rejects_ontology(use_api):
    client = CogneeClient.__new__(CogneeClient)
    client.use_api = use_api
    with pytest.raises(ValueError, match="permanent memory"):
        await client.remember("hello", session_id="session", ontology_key="one")


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_local_resolves_for_default_user(monkeypatch, missing):
    from importlib import import_module

    import cognee.modules.users.methods as users
    from cognee.api.v1.ontologies.ontologies import OntologyService

    resolver_module = import_module("cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver")
    user = SimpleNamespace(id="local-user")
    monkeypatch.setattr(users, "get_default_user", AsyncMock(return_value=user))
    lookup = Mock(return_value=["first OWL", "second OWL"])
    if missing:
        lookup.side_effect = ValueError("Ontology key 'one' not found")
    monkeypatch.setattr(OntologyService, "get_ontology_contents", lookup)
    resolver = Mock()
    monkeypatch.setattr(resolver_module, "RDFLibOntologyResolver", resolver)
    client = CogneeClient.__new__(CogneeClient)
    client.use_api = False
    client.cognee = SimpleNamespace(remember=AsyncMock())
    if missing:
        with pytest.raises(ValueError, match="not found"):
            await client.remember("hello", ontology_key=["one", "two"])
        client.cognee.remember.assert_not_awaited()
    else:
        await client.remember("hello", ontology_key=["one", "two"])
        streams = resolver.call_args.kwargs["ontology_file"]
        assert [s.read() for s in streams] == ["first OWL", "second OWL"]
        kwargs = client.cognee.remember.call_args.kwargs
        assert kwargs["config"]["ontology_config"]["ontology_resolver"] is resolver.return_value
        # The user is deliberately NOT pinned on the call: remember() resolves it to
        # get_default_user() itself, so passing it only when an ontology was requested
        # would give the same tool call two user-resolution paths. The lookup below is
        # what has to agree with the write, and it does.
        assert "user" not in kwargs
    lookup.assert_called_once_with(["one", "two"], user)


@pytest.mark.asyncio
async def test_local_omission_preserves_config():
    client = CogneeClient.__new__(CogneeClient)
    client.use_api = False
    client.cognee = SimpleNamespace(remember=AsyncMock())
    await client.remember("hello")
    client.cognee.remember.assert_awaited_once_with(data="hello", dataset_name="main_dataset")


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("keys", ["one", ["one", "two"]])
async def test_mcp_forwards_ontology(monkeypatch, background, keys):
    remember = AsyncMock(return_value={"status": "completed"})
    monkeypatch.setattr(server, "cognee_client", SimpleNamespace(remember=remember))
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "remember")
        assert "ontology_key" in tool.inputSchema["properties"]
        result = await client.call_tool(
            "remember", {"data": "hello", "ontology_key": keys, "background": background}
        )
        assert not result.is_error
        await asyncio.sleep(0)
    remember.assert_awaited_once()
    assert remember.call_args.kwargs["ontology_key"] == keys
