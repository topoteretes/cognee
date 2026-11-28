"""Recall forwards retriever_specific_config, and server-side defaults fill in what the caller leaves out."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastmcp import Client
from src import server
from src.cognee_client import CogneeClient

DEFAULT_ENV = (
    "COGNEE_MCP_RECALL_SEARCH_TYPE",
    "COGNEE_MCP_RECALL_TOP_K",
    "COGNEE_MCP_RECALL_RETRIEVER_CONFIG",
    "COGNEE_MCP_RECALL_RETRIEVER_CONFIG_FILE",
    "COGNEE_MCP_RECALL_SYSTEM_PROMPT",
    "COGNEE_MCP_RECALL_SYSTEM_PROMPT_FILE",
)


@pytest.fixture(autouse=True)
def _no_recall_defaults(monkeypatch):
    for name in DEFAULT_ENV:
        monkeypatch.delenv(name, raising=False)


async def _api_payloads(calls: list[dict]) -> list[dict]:
    """Run recall once per kwargs dict against a mock API; return the JSON bodies sent."""
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        for kwargs in calls:
            await client.recall("q", datasets=["d"], **kwargs)
    finally:
        await client.close()
    return [json.loads(r.content.decode()) for r in requests]


# --------------------------------------------------------------------------- the tool


@pytest.mark.asyncio
async def test_recall_tool_forwards_retriever_specific_config(monkeypatch):
    fake = SimpleNamespace(recall=AsyncMock(return_value=[]), get_recall_state=AsyncMock())
    monkeypatch.setattr(server, "cognee_client", fake)
    config = {"max_subqueries": 4, "chunks_top_k": 20}

    await server.recall(
        "q",
        datasets="d",
        search_type="HYBRID_COMPLETION_DECOMPOSITION",
        retriever_specific_config=config,
    )
    assert fake.recall.call_args.kwargs["retriever_specific_config"] == config

    await server.recall("q", datasets="d")
    assert fake.recall.call_args.kwargs["retriever_specific_config"] is None
    # top_k is left to the client's default when the caller sends none
    assert fake.recall.call_args.kwargs["top_k"] is None


@pytest.mark.asyncio
async def test_recall_tool_rejects_non_dict_retriever_specific_config():
    result = await server.recall("q", retriever_specific_config="max_subqueries=4")
    assert result[0].text.startswith("Error:")
    assert "retriever_specific_config must be a JSON object" in result[0].text


@pytest.mark.asyncio
async def test_recall_tool_still_validates_an_explicit_top_k():
    result = await server.recall("q", top_k=0)
    assert result[0].text.startswith("Error:")
    assert "top_k must be at least 1" in result[0].text


@pytest.mark.asyncio
async def test_recall_tool_schema_exposes_retriever_specific_config():
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    props = tools["recall"].inputSchema.get("properties", {})
    assert "retriever_specific_config" in props
    assert "max_subqueries" in props["retriever_specific_config"].get("description", "")
    # the model picks search types from this description, so the decomposition type must be listed
    assert "HYBRID_COMPLETION_DECOMPOSITION" in props["search_type"].get("description", "")


# --------------------------------------------------------------------------- the client


@pytest.mark.asyncio
async def test_api_client_sends_retriever_specific_config_only_when_set():
    sent = await _api_payloads([{"retriever_specific_config": {"max_subqueries": 4}}, {}])
    assert sent[0]["retriever_specific_config"] == {"max_subqueries": 4}
    assert "retriever_specific_config" not in sent[1]
    assert sent[1]["top_k"] == 15  # unchanged default when nothing is configured
    assert sent[1]["search_type"] is None  # auto-routing stays on


@pytest.mark.asyncio
async def test_direct_client_passes_retriever_specific_config_only_when_set():
    client = CogneeClient()
    client.cognee = SimpleNamespace(recall=AsyncMock(return_value=[]))

    await client.recall("q", datasets=["d"], retriever_specific_config={"facts_top_k": 20})
    assert client.cognee.recall.call_args.kwargs["retriever_specific_config"] == {"facts_top_k": 20}

    await client.recall("q", datasets=["d"])
    assert "retriever_specific_config" not in client.cognee.recall.call_args.kwargs
    assert client.cognee.recall.call_args.kwargs["top_k"] == 15


@pytest.mark.asyncio
async def test_server_defaults_fill_in_what_the_caller_leaves_out(monkeypatch, tmp_path):
    (tmp_path / "plan.txt").write_text("Plan the searches.\n", encoding="utf-8")
    (tmp_path / "recall.json").write_text(
        json.dumps(
            {
                "max_subqueries": 4,
                "chunks_top_k": 20,
                "decomposition_system_prompt_file": "plan.txt",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COGNEE_MCP_RECALL_SEARCH_TYPE", "hybrid_completion_decomposition")
    monkeypatch.setenv("COGNEE_MCP_RECALL_TOP_K", "20")
    monkeypatch.setenv("COGNEE_MCP_RECALL_RETRIEVER_CONFIG_FILE", str(tmp_path / "recall.json"))

    sent = await _api_payloads(
        [
            {},
            {
                "search_type": "CHUNKS",
                "top_k": 5,
                "retriever_specific_config": {"max_subqueries": 2},
            },
        ]
    )

    # nothing sent: every default applies, the planning prompt is read from its file
    assert sent[0]["search_type"] == "HYBRID_COMPLETION_DECOMPOSITION"
    assert sent[0]["top_k"] == 20
    assert sent[0]["retriever_specific_config"] == {
        "max_subqueries": 4,
        "chunks_top_k": 20,
        "decomposition_system_prompt": "Plan the searches.",
    }
    # values the caller sends win; the other configured keys stay
    assert sent[1]["search_type"] == "CHUNKS"
    assert sent[1]["top_k"] == 5
    assert sent[1]["retriever_specific_config"]["max_subqueries"] == 2
    assert sent[1]["retriever_specific_config"]["chunks_top_k"] == 20


@pytest.mark.asyncio
async def test_inline_retriever_config_default(monkeypatch):
    monkeypatch.setenv("COGNEE_MCP_RECALL_RETRIEVER_CONFIG", '{"facts_top_k": 20}')
    sent = await _api_payloads([{}])
    assert sent[0]["retriever_specific_config"] == {"facts_top_k": 20}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("COGNEE_MCP_RECALL_RETRIEVER_CONFIG", "{not json"),
        ("COGNEE_MCP_RECALL_RETRIEVER_CONFIG", "[1, 2]"),
        ("COGNEE_MCP_RECALL_RETRIEVER_CONFIG_FILE", "/does/not/exist.json"),
        ("COGNEE_MCP_RECALL_TOP_K", "many"),
        ("COGNEE_MCP_RECALL_TOP_K", "0"),
    ],
)
async def test_unusable_defaults_are_ignored(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    sent = await _api_payloads([{}])
    assert "retriever_specific_config" not in sent[0]
    assert sent[0]["top_k"] == 15
