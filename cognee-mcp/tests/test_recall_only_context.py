"""Recall can return the retrieved context instead of the answer."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from src import server
from src.cognee_client import CogneeClient


@pytest.mark.asyncio
async def test_recall_tool_forwards_only_context(monkeypatch):
    fake = SimpleNamespace(
        recall=AsyncMock(return_value=[{"source": "graph", "text": "a retrieved passage"}]),
        get_recall_state=AsyncMock(),
    )
    monkeypatch.setattr(server, "cognee_client", fake)

    result = await server.recall("q", search_type="GRAPH_COMPLETION", only_context=True)
    assert fake.recall.call_args.kwargs["only_context"] is True
    assert "a retrieved passage" in result[0].text

    await server.recall("q", search_type="GRAPH_COMPLETION")
    assert fake.recall.call_args.kwargs["only_context"] is False


@pytest.mark.asyncio
async def test_direct_client_passes_only_context_only_when_set():
    client = CogneeClient()
    client.cognee = SimpleNamespace(recall=AsyncMock(return_value=[]))

    await client.recall("q", datasets=["d"], system_prompt="s", only_context=True)
    assert client.cognee.recall.call_args.kwargs["only_context"] is True

    await client.recall("q", datasets=["d"], system_prompt="s")
    assert "only_context" not in client.cognee.recall.call_args.kwargs


@pytest.mark.asyncio
async def test_api_client_sends_only_context_only_when_set():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.recall("q", datasets=["d"], system_prompt="s", only_context=True)
        await client.recall("q", datasets=["d"], system_prompt="s")
    finally:
        await client.close()

    assert json.loads(requests[0].content.decode())["only_context"] is True
    assert "only_context" not in json.loads(requests[1].content.decode())


HYBRID_CONTEXT = (
    "The question is: `Who gets education leads?`\n"
    "Answer using this sectioned context. Keep the answer brief and do not use "
    "information outside the context.\n\nContext:\n`## Relevant passages\nEducation leads go to Fares.`"
)


@pytest.mark.asyncio
async def test_only_context_leaves_out_the_brief_answer_instruction(monkeypatch):
    fake = SimpleNamespace(
        recall=AsyncMock(return_value=[{"source": "graph", "text": HYBRID_CONTEXT}]),
        get_recall_state=AsyncMock(),
    )
    monkeypatch.setattr(server, "cognee_client", fake)

    result = await server.recall("q", search_type="HYBRID_COMPLETION", only_context=True)
    text = result[0].text
    assert "Keep the answer brief" not in text
    assert (
        "Answer using this sectioned context. Do not use information outside the context." in text
    )
    assert "Education leads go to Fares." in text


@pytest.mark.asyncio
async def test_answers_are_returned_unchanged_without_only_context(monkeypatch):
    # The clause only ever appears in a returned context; an answer is passed through as is.
    fake = SimpleNamespace(
        recall=AsyncMock(return_value=[{"source": "graph", "text": HYBRID_CONTEXT}]),
        get_recall_state=AsyncMock(),
    )
    monkeypatch.setattr(server, "cognee_client", fake)

    result = await server.recall("q", search_type="HYBRID_COMPLETION")
    assert "Keep the answer brief" in result[0].text
