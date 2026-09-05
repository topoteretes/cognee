"""MCP improve / remember: the same improve options as the SDK, summarized per stage.

``improve`` is not an advertised MCP tool (the LLM-facing surface is pinned to
remember / recall / forget), but it is the coroutine an operator or a future
registration reaches, and ``remember`` runs the same loop through
``self_improvement``. These tests pin what both forward to ``CogneeClient``
and what the client forwards to the SDK or the HTTP API.
"""

import importlib
import json
import sys
from pathlib import Path

import httpx
import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

CogneeClient = importlib.import_module("src.cognee_client").CogneeClient


def _improve_payload(**overrides):
    payload = {
        "dataset_id": "00000000-0000-0000-0000-000000000001",
        "dataset_name": "ds",
        "session_ids": ["s1"],
        "background": False,
        "finished": True,
        "error": None,
        "memify_run": {},
        "status": "completed",
        "stages": [
            {"stage": "feedback_weights", "status": "skipped", "reason": "no_session_ids"},
            {"stage": "triplet_enrichment", "status": "completed", "counts": {"nodes": 3}},
        ],
    }
    payload.update(overrides)
    return payload


class RecordingImproveClient:
    def __init__(self, result=None):
        self.calls: list[dict] = []
        self._result = result if result is not None else _improve_payload()

    async def improve(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


# --- server.improve -------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_improve_forwards_every_option(monkeypatch):
    import src.server as server

    fake_client = RecordingImproveClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    await server.improve(
        dataset_name="ds",
        session_ids="s1, s2",
        node_name="Alice,Bob",
        build_global_context_index=True,
        build_truth_subspace=True,
    )

    assert fake_client.calls == [
        {
            "dataset_name": "ds",
            "session_ids": ["s1", "s2"],
            "node_name": ["Alice", "Bob"],
            "build_global_context_index": True,
            "build_truth_subspace": True,
        }
    ]


@pytest.mark.asyncio
async def test_mcp_improve_summarizes_every_stage(monkeypatch):
    import src.server as server

    monkeypatch.setattr(server, "cognee_client", RecordingImproveClient())

    result = await server.improve(dataset_name="ds")

    text = result[0].text
    assert text.splitlines()[0] == "Improve completed for dataset 'ds'. Sessions: 1."
    assert "- feedback_weights: skipped (no_session_ids)" in text
    assert "- triplet_enrichment: completed (nodes=3)" in text


@pytest.mark.asyncio
async def test_mcp_improve_reports_errored_stages_and_failures(monkeypatch):
    import src.server as server

    payload = _improve_payload(
        status="errored",
        stages=[
            {
                "stage": "persist_agent_traces",
                "status": "errored",
                "error": "RuntimeError: boom",
            }
        ],
    )
    monkeypatch.setattr(server, "cognee_client", RecordingImproveClient(payload))
    text = (await server.improve(dataset_name="ds"))[0].text
    assert "Improve errored" in text
    assert "- persist_agent_traces: errored (RuntimeError: boom)" in text

    class ExplodingClient:
        async def improve(self, **kwargs):
            raise RuntimeError("upstream is down")

    monkeypatch.setattr(server, "cognee_client", ExplodingClient())
    text = (await server.improve(dataset_name="ds"))[0].text
    assert text.startswith("Error: Improve failed")
    assert "upstream is down" in text


def test_mcp_improve_docstring_no_longer_promises_a_session_sync():
    import src.server as server

    doc = (server.improve.__doc__ or "").lower()
    assert "sync" not in doc
    assert "build_truth_subspace" in doc
    assert "node_name" in doc


def test_format_improve_result_handles_running_and_legacy_payloads():
    import src.server as server

    running = server.format_improve_result(
        _improve_payload(status="running", stages=[], finished=False), "ds"
    )
    assert "running" in running.splitlines()[0]
    assert "background" in running

    legacy = server.format_improve_result({"some-uuid": {"status": "completed"}}, "ds")
    assert legacy.startswith("Improve completed for dataset 'ds'.")


# --- CogneeClient.improve --------------------------------------------------------


@pytest.mark.asyncio
async def test_cognee_client_api_improve_sends_every_option():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_improve_payload())

    client = CogneeClient(api_url="http://cognee.local", api_token="token")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        result = await client.improve(
            dataset_name="ds",
            session_ids=["s1"],
            node_name=["Alice"],
            build_global_context_index=True,
            build_truth_subspace=True,
        )
    finally:
        await client.close()

    assert requests[0].url.path == "/api/v1/improve"
    payload = json.loads(requests[0].content.decode())
    assert payload == {
        "dataset_name": "ds",
        "session_ids": ["s1"],
        "node_name": ["Alice"],
        "build_global_context_index": True,
        "build_truth_subspace": True,
    }
    assert result["status"] == "completed"
    assert [stage["stage"] for stage in result["stages"]] == [
        "feedback_weights",
        "triplet_enrichment",
    ]


@pytest.mark.asyncio
async def test_cognee_client_api_improve_omits_defaults():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_improve_payload())

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.improve(dataset_name="ds")
    finally:
        await client.close()

    assert json.loads(requests[0].content.decode()) == {"dataset_name": "ds"}


@pytest.mark.asyncio
async def test_cognee_client_local_improve_returns_the_result_as_json_dict():
    from cognee.modules.improve import ImproveResult, StageResult

    class FakeCognee:
        def __init__(self):
            self.calls = []

        async def improve(self, **kwargs):
            self.calls.append(kwargs)
            return ImproveResult(
                dataset_name="ds",
                stages=[StageResult.skipped("global_context_index", "opt_in_disabled")],
                memify_run={},
            )

    fake = FakeCognee()
    client = CogneeClient()
    client.cognee = fake

    result = await client.improve(
        dataset_name="ds",
        session_ids=["s1"],
        node_name=["Alice"],
        build_global_context_index=True,
        build_truth_subspace=True,
    )

    assert fake.calls == [
        {
            "dataset": "ds",
            "session_ids": ["s1"],
            "node_name": ["Alice"],
            "build_global_context_index": True,
            "build_truth_subspace": True,
        }
    ]
    assert result["status"] == "skipped"
    assert result["stages"][0]["reason"] == "opt_in_disabled"


# --- remember(self_improvement=...) ----------------------------------------------


class RecordingRememberClient:
    def __init__(self):
        self.calls: list[dict] = []

    async def remember(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "completed"}


@pytest.mark.asyncio
async def test_mcp_remember_forwards_self_improvement(monkeypatch):
    import src.server as server

    fake_client = RecordingRememberClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    await server.remember(data="alice prefers async updates", dataset_name="ds")
    await server.remember(
        data="alice prefers async updates", dataset_name="ds", self_improvement=False
    )

    assert fake_client.calls[0]["self_improvement"] is True
    assert fake_client.calls[1]["self_improvement"] is False


@pytest.mark.asyncio
async def test_mcp_remember_background_forwards_self_improvement(monkeypatch):
    import src.server as server

    fake_client = RecordingRememberClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)
    tracked = []

    def track(coro):
        tracked.append(coro)

    monkeypatch.setattr(server, "_track_background", track)

    await server.remember(data="note", dataset_name="ds", background=True, self_improvement=False)

    assert len(tracked) == 1
    await tracked[0]
    assert fake_client.calls[0]["self_improvement"] is False


@pytest.mark.asyncio
async def test_mcp_remember_advertises_self_improvement():
    import src.server as server

    tools = await server.mcp.list_tools()
    remember_tool = next(tool for tool in tools if tool.name == "remember")

    assert "self_improvement" in remember_tool.parameters["properties"]
    assert remember_tool.parameters["properties"]["self_improvement"]["default"] is True


@pytest.mark.asyncio
async def test_cognee_client_local_remember_forwards_self_improvement_false_only():
    class FakeCognee:
        def __init__(self):
            self.calls = []

        async def remember(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "completed"}

    fake = FakeCognee()
    client = CogneeClient()
    client.cognee = fake

    await client.remember(data="text", dataset_name="ds")
    await client.remember(data="text", dataset_name="ds", self_improvement=False)

    # The SDK default is True, so only an explicit opt-out travels.
    assert "self_improvement" not in fake.calls[0]
    assert fake.calls[1]["self_improvement"] is False


@pytest.mark.asyncio
async def test_cognee_client_api_remember_sends_self_improvement_false():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.remember("hello", dataset_name="ds")
        await client.remember("hello", dataset_name="ds", self_improvement=False)
    finally:
        await client.close()

    assert requests[0].url.path == "/api/v1/remember"
    assert b'name="self_improvement"' not in requests[0].content
    assert b'name="self_improvement"' in requests[1].content
    assert b"false" in requests[1].content
