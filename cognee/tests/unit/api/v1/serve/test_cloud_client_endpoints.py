"""CloudClient's dataset, agent-connection, and user endpoints.

These are the endpoint groups every agent integration calls over raw
HTTP today: datasets (name→id resolution, idempotent ensure, background
pipeline polling), the agents connection lifecycle, and principal
resolution via users/me.
"""

import asyncio
from uuid import UUID

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.api.v1.serve.exceptions import CogneeClientRequestError


class FakeResponse:
    def __init__(self, status=200, json_body=None, text_body=""):
        self.status = status
        self._json = json_body if json_body is not None else {}
        self._text = text_body

    async def json(self):
        return self._json

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    closed = False

    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.calls = []

    def _record(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.response

    def get(self, url, **kwargs):
        return self._record("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._record("POST", url, **kwargs)

    def delete(self, url, **kwargs):
        return self._record("DELETE", url, **kwargs)


def make_client(response=None):
    client = CloudClient("http://cloud.example", "ck_test")
    session = FakeSession(response)
    client._session = session
    return client, session


# ----- datasets -----


def test_datasets_list_hits_collection_route():
    client, session = make_client(FakeResponse(json_body=[]))
    asyncio.run(client.datasets_list())
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://cloud.example/api/v1/datasets"


def test_datasets_create_posts_name():
    client, session = make_client(FakeResponse(json_body={"id": "d1", "name": "agent_sessions"}))
    result = asyncio.run(client.datasets_create("agent_sessions"))
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["json"] == {"name": "agent_sessions"}
    assert result["name"] == "agent_sessions"


def test_datasets_status_sends_repeated_query_params():
    client, session = make_client(FakeResponse(json_body={}))
    asyncio.run(
        client.datasets_status(
            ["11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"],
            pipelines=["cognify_pipeline", "memify_pipeline"],
        )
    )
    call = session.calls[0]
    assert call["url"].endswith("/api/v1/datasets/status")
    assert call["params"] == [
        ("dataset", "11111111-1111-1111-1111-111111111111"),
        ("dataset", "22222222-2222-2222-2222-222222222222"),
        ("pipeline", "cognify_pipeline"),
        ("pipeline", "memify_pipeline"),
    ]


def test_datasets_data_and_delete_target_dataset_routes():
    dataset_id = UUID("00000000-0000-0000-0000-000000000042")

    client, session = make_client(FakeResponse(json_body=[]))
    asyncio.run(client.datasets_data(dataset_id))
    assert session.calls[0]["url"].endswith(f"/api/v1/datasets/{dataset_id}/data")

    client, session = make_client(FakeResponse(status=204))
    asyncio.run(client.datasets_delete(dataset_id))
    call = session.calls[0]
    assert call["method"] == "DELETE"
    assert call["url"].endswith(f"/api/v1/datasets/{dataset_id}")


def test_delete_with_empty_body_returns_none():
    client, _ = make_client(FakeResponse(status=200, json_body=None, text_body=""))

    class _NoJson(FakeResponse):
        async def json(self):
            raise ValueError("no body")

    client, _ = make_client(_NoJson(status=200))
    assert asyncio.run(client.datasets_delete_all()) is None


# ----- agent connections -----


def test_agents_register_composes_connection_payload():
    client, session = make_client()
    asyncio.run(
        client.agents_register(
            "claude-code",
            memory_mode="hybrid",
            session_id="session-1",
            dataset_names=["agent_sessions"],
            origin_function="session_start",
            metadata={"host": "claude"},
        )
    )
    payload = session.calls[0]["json"]
    assert payload == {
        "agent_session_name": "claude-code",
        "type": "api",
        "memory_mode": "hybrid",
        "source": "api",
        "session_id": "session-1",
        "dataset_names": ["agent_sessions"],
        "origin_function": "session_start",
        "metadata": {"host": "claude"},
    }
    assert session.calls[0]["url"].endswith("/api/v1/agents/register")


def test_agents_unregister_sends_connection_name():
    client, session = make_client()
    asyncio.run(client.agents_unregister("claude-code"))
    assert session.calls[0]["json"] == {"agent_session_name": "claude-code"}
    assert session.calls[0]["url"].endswith("/api/v1/agents/unregister")


def test_agents_connections_me_filters_by_name():
    client, session = make_client()
    asyncio.run(client.agents_connections_me("claude-code"))
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"].endswith("/api/v1/agents/connections/me")
    assert call["params"] == {"agent_session_name": "claude-code"}


def test_agents_connections_me_404_raises_typed_error():
    client, _ = make_client(FakeResponse(status=404, text_body="connection not found"))
    with pytest.raises(CogneeClientRequestError) as excinfo:
        asyncio.run(client.agents_connections_me())
    assert excinfo.value.status == 404
    assert excinfo.value.operation == "agents_connections_me"


# ----- users -----


def test_users_me_resolves_principal():
    client, session = make_client(FakeResponse(json_body={"id": "u1", "email": "a@b.c"}))
    result = asyncio.run(client.users_me())
    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["url"].endswith("/api/v1/users/me")
    assert result["email"] == "a@b.c"
