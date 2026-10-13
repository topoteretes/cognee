"""Dataset ids on the MCP memory tools.

Cognee shares datasets across users only by id: a dataset name resolves to a
dataset the caller owns. These tests cover the recall/remember tools accepting
dataset ids, forwarding them unchanged to the REST API or the SDK, and leaving
name-only calls exactly as they were.
"""

import importlib
import json
import sys
from pathlib import Path
from uuid import UUID

import httpx
import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

CogneeClient = importlib.import_module("src.cognee_client").CogneeClient

SHARED_DATASET_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_DATASET_ID = UUID("22222222-2222-2222-2222-222222222222")


async def _mock_api_client(requests: list[httpx.Request], response_json) -> CogneeClient:
    """API-mode client whose requests are recorded instead of sent over the wire."""

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_json)

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


class FakeCogneeModule:
    """Stand-in for the `cognee` module used by CogneeClient's direct mode."""

    def __init__(self):
        self.remember_calls: list[dict] = []
        self.recall_calls: list[dict] = []

    async def remember(self, **kwargs):
        self.remember_calls.append(kwargs)

    async def recall(self, **kwargs):
        self.recall_calls.append(kwargs)
        return [{"text": "ok"}]


def _local_client(fake_cognee: FakeCogneeModule) -> CogneeClient:
    """Direct-mode client with the real cognee module swapped out."""
    client = CogneeClient()
    client.cognee = fake_cognee
    return client


# --- CogneeClient, API mode ---------------------------------------------------


@pytest.mark.asyncio
async def test_cognee_client_api_recall_sends_dataset_ids():
    """Ids go to /api/v1/recall as dataset_ids, with no fallback dataset listing."""
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests, response_json=[])

    try:
        await client.recall("hello", dataset_ids=[SHARED_DATASET_ID, OTHER_DATASET_ID])
    finally:
        await client.close()

    # One request: an id-scoped recall must not list datasets to fill in names.
    assert [request.url.path for request in requests] == ["/api/v1/recall"]
    payload = json.loads(requests[0].content.decode())
    assert payload["dataset_ids"] == [str(SHARED_DATASET_ID), str(OTHER_DATASET_ID)]
    assert "datasets" not in payload


@pytest.mark.asyncio
async def test_cognee_client_api_recall_without_dataset_ids_is_unchanged():
    """A name-scoped recall sends no dataset_ids field."""
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests, response_json=[])

    try:
        await client.recall("hello", datasets=["docs"])
    finally:
        await client.close()

    payload = json.loads(requests[0].content.decode())
    assert payload["datasets"] == ["docs"]
    assert "dataset_ids" not in payload


@pytest.mark.asyncio
async def test_cognee_client_api_remember_sends_dataset_id_form_field():
    """Permanent remember targets the dataset with the multipart datasetId field."""
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests, response_json={"status": "completed"})

    try:
        await client.remember(
            "alice prefers async updates", dataset_name=None, dataset_id=SHARED_DATASET_ID
        )
    finally:
        await client.close()

    assert requests[0].url.path == "/api/v1/remember"
    body = requests[0].content.decode("latin-1")
    assert 'name="datasetId"' in body
    assert str(SHARED_DATASET_ID) in body
    assert 'name="datasetName"' not in body


@pytest.mark.asyncio
async def test_cognee_client_api_remember_without_dataset_id_is_unchanged():
    """A name-only remember still sends datasetName and no datasetId."""
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests, response_json={"status": "completed"})

    try:
        await client.remember("alice prefers async updates", dataset_name="ds")
    finally:
        await client.close()

    body = requests[0].content.decode("latin-1")
    assert 'name="datasetName"' in body
    assert 'name="datasetId"' not in body


@pytest.mark.asyncio
async def test_cognee_client_api_session_remember_sends_dataset_id():
    """Session entries carry dataset_id in the /remember/entry JSON body."""
    requests: list[httpx.Request] = []
    client = await _mock_api_client(requests, response_json={"status": "completed"})

    try:
        await client.remember(
            "scratch note",
            dataset_name=None,
            session_id="session-1",
            dataset_id=SHARED_DATASET_ID,
        )
    finally:
        await client.close()

    assert requests[0].url.path == "/api/v1/remember/entry"
    payload = json.loads(requests[0].content.decode())
    assert payload["dataset_id"] == str(SHARED_DATASET_ID)
    assert "dataset_name" not in payload
    assert payload["session_id"] == "session-1"


# --- CogneeClient, direct mode ------------------------------------------------


@pytest.mark.asyncio
async def test_cognee_client_local_remember_forwards_dataset_id():
    """Direct mode hands the id to cognee.remember as a UUID."""
    fake = FakeCogneeModule()
    client = _local_client(fake)

    result = await client.remember(data="hello", dataset_name=None, dataset_id=SHARED_DATASET_ID)

    assert fake.remember_calls[0]["dataset_id"] == SHARED_DATASET_ID
    assert "dataset_name" not in fake.remember_calls[0]
    assert result["dataset_id"] == str(SHARED_DATASET_ID)


@pytest.mark.asyncio
async def test_cognee_client_local_recall_forwards_dataset_ids():
    """Direct mode hands the ids to cognee.recall."""
    fake = FakeCogneeModule()
    client = _local_client(fake)

    await client.recall("hello", dataset_ids=[SHARED_DATASET_ID])

    assert fake.recall_calls[0]["dataset_ids"] == [SHARED_DATASET_ID]
    assert "datasets" not in fake.recall_calls[0]


# --- MCP tools ----------------------------------------------------------------


class RecordingClient:
    """Fake cognee_client that records what the tools forwarded."""

    def __init__(self):
        self.remember_calls: list[dict] = []
        self.recall_calls: list[dict] = []
        self.forget_calls: list[dict] = []

    async def remember(self, **kwargs):
        self.remember_calls.append(kwargs)
        return {"status": "completed"}

    async def recall(self, **kwargs):
        self.recall_calls.append(kwargs)
        return [{"text": "ok"}]

    async def forget(self, **kwargs):
        self.forget_calls.append(kwargs)
        return {"status": "success"}


@pytest.mark.asyncio
async def test_mcp_recall_parses_and_forwards_dataset_ids(monkeypatch):
    from src import server

    fake_client = RecordingClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    result = await server.recall(
        query="hello",
        dataset_ids=f"{SHARED_DATASET_ID}, {OTHER_DATASET_ID}",
    )

    assert "ok" in result[0].text
    assert fake_client.recall_calls[0]["dataset_ids"] == [SHARED_DATASET_ID, OTHER_DATASET_ID]


@pytest.mark.asyncio
async def test_mcp_recall_rejects_malformed_dataset_ids(monkeypatch):
    from src import server

    fake_client = RecordingClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    result = await server.recall(query="hello", dataset_ids="not-a-uuid")

    assert result[0].text.startswith("Error: invalid UUID")
    assert fake_client.recall_calls == []


@pytest.mark.asyncio
async def test_mcp_remember_forwards_dataset_id_instead_of_agent_default(monkeypatch):
    """An explicit id is the target, so the agent-scoped default name is not applied."""
    from src import server

    fake_client = RecordingClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)
    monkeypatch.setattr(server, "_agent_scoped_default_dataset", lambda: "cursor_vscode_memory")

    result = await server.remember(
        data="alice prefers async updates", dataset_id=str(SHARED_DATASET_ID)
    )

    assert fake_client.remember_calls[0]["dataset_id"] == SHARED_DATASET_ID
    assert fake_client.remember_calls[0]["dataset_name"] is None
    assert str(SHARED_DATASET_ID) in result[0].text


@pytest.mark.asyncio
async def test_mcp_remember_without_dataset_id_is_unchanged(monkeypatch):
    """Name-only calls keep the agent-scoped default and forward no dataset_id."""
    from src import server

    fake_client = RecordingClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)
    monkeypatch.setattr(server, "_agent_scoped_default_dataset", lambda: "cursor_vscode_memory")

    await server.remember(data="alice prefers async updates")

    assert fake_client.remember_calls[0]["dataset_name"] == "cursor_vscode_memory"
    assert "dataset_id" not in fake_client.remember_calls[0]


@pytest.mark.asyncio
async def test_mcp_remember_rejects_malformed_dataset_id(monkeypatch):
    from src import server

    fake_client = RecordingClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    result = await server.remember(data="hello", dataset_id="not-a-uuid")

    assert result[0].text.startswith("Error: invalid UUID")
    assert fake_client.remember_calls == []


@pytest.mark.asyncio
async def test_mcp_forget_forwards_dataset_id(monkeypatch):
    """forget already accepts a dataset id; pin its forwarding alongside the others."""
    from src import server

    fake_client = RecordingClient()
    monkeypatch.setattr(server, "cognee_client", fake_client)

    result = await server.forget(dataset_id=str(SHARED_DATASET_ID))

    assert fake_client.forget_calls[0]["dataset_id"] == SHARED_DATASET_ID
    assert str(SHARED_DATASET_ID) in result[0].text
