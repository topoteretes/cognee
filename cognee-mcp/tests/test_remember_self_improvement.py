"""Tests for exposing self_improvement through the MCP remember path.

remember() self-improves by default, and improve() reads the whole graph — on a
large one that dominates the cost of the write. The MCP tool used to build its
kwargs without the flag, so an MCP deployment had no way to opt out.
"""

import importlib
import sys
import types
from pathlib import Path

import httpx
import pytest


MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

CogneeClient = importlib.import_module("src.cognee_client").CogneeClient


def _direct_client_with_spy():
    """A direct-mode client whose cognee.remember records its kwargs."""
    client = CogneeClient()
    calls = []

    async def fake_remember(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(status="completed")

    client.cognee = types.SimpleNamespace(remember=fake_remember)
    return client, calls


@pytest.mark.asyncio
async def test_direct_mode_forwards_self_improvement_false():
    client, calls = _direct_client_with_spy()

    await client.remember("a fact", dataset_name="ds", self_improvement=False)

    assert calls[0]["self_improvement"] is False


@pytest.mark.asyncio
async def test_direct_mode_forwards_self_improvement_true():
    client, calls = _direct_client_with_spy()

    await client.remember("a fact", dataset_name="ds", self_improvement=True)

    assert calls[0]["self_improvement"] is True


@pytest.mark.asyncio
async def test_direct_mode_omits_self_improvement_when_unset():
    """None must leave remember()'s own default in place, not force a value."""
    client, calls = _direct_client_with_spy()

    await client.remember("a fact", dataset_name="ds")

    assert "self_improvement" not in calls[0]


@pytest.mark.asyncio
async def test_api_mode_sends_self_improvement_form_field():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.remember("a fact", dataset_name="ds", self_improvement=False)
    finally:
        await client.close()

    assert requests[0].url.path == "/api/v1/remember"
    body = requests[0].content.decode()
    assert 'name="self_improvement"' in body
    assert "false" in body


@pytest.mark.asyncio
async def test_api_mode_omits_self_improvement_when_unset():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.remember("a fact", dataset_name="ds")
    finally:
        await client.close()

    assert 'name="self_improvement"' not in requests[0].content.decode()


def test_env_var_default_is_on_unless_disabled(monkeypatch):
    server = importlib.import_module("src.server")

    monkeypatch.delenv("COGNEE_MCP_REMEMBER_SELF_IMPROVEMENT", raising=False)
    assert server._default_remember_self_improvement() is True

    monkeypatch.setenv("COGNEE_MCP_REMEMBER_SELF_IMPROVEMENT", "false")
    assert server._default_remember_self_improvement() is False

    monkeypatch.setenv("COGNEE_MCP_REMEMBER_SELF_IMPROVEMENT", "FALSE")
    assert server._default_remember_self_improvement() is False

    monkeypatch.setenv("COGNEE_MCP_REMEMBER_SELF_IMPROVEMENT", "true")
    assert server._default_remember_self_improvement() is True


def test_remember_tool_exposes_self_improvement():
    """The tool signature is what the LLM sees — the flag has to be on it."""
    import inspect

    server = importlib.import_module("src.server")

    signature = inspect.signature(server.remember)

    assert "self_improvement" in signature.parameters
