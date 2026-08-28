"""The SDK routing sites must deliver the callers' params to the remote client.

Client-side forwarding (test_cloud_client_forwarding) is only half the path:
the module-level functions decide *what* reaches the client. These tests
pin the routing-site contracts: improve() hands over ``session_ids`` and
``run_in_background`` (hermes-agent's documented reason for bypassing
CloudClient), add() hands over ``node_set``/``dataset_id``, recall() maps
``auto_route`` onto the tri-state ``search_type``, and search() forwards
``session_id``.
"""

import asyncio
from uuid import UUID

import pytest

import cognee
from cognee.api.v1.serve import state
from cognee.api.v1.serve.state import UNSET, set_remote_client
from cognee.modules.search.types import SearchType


class StubClient:
    def __init__(self):
        self.calls = []

    async def improve(self, dataset, **kwargs):
        self.calls.append(("improve", dataset, kwargs))
        return {}

    async def remember(self, data, dataset_name, **kwargs):
        self.calls.append(("remember", data, dataset_name, kwargs))
        return {"status": "completed", "dataset_name": dataset_name}

    async def add(self, data, dataset_name, **kwargs):
        self.calls.append(("add", data, dataset_name, kwargs))
        return {"status": "ok"}

    async def recall(self, query_text, query_type=UNSET, **kwargs):
        self.calls.append(("recall", query_text, query_type, kwargs))
        return []

    async def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return []


@pytest.fixture
def stub_client(monkeypatch):
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    stub = StubClient()
    set_remote_client(stub)
    yield stub
    set_remote_client(None)


def last_call(stub, operation):
    calls = [call for call in stub.calls if call[0] == operation]
    assert calls, f"no {operation} call reached the remote client"
    return calls[-1]


def test_improve_routes_session_ids_and_background(stub_client):
    asyncio.run(
        cognee.improve(
            "agent_sessions",
            session_ids=["chat_1", "chat_2"],
            run_in_background=True,
            node_name=["entity"],
        )
    )
    _, dataset, kwargs = last_call(stub_client, "improve")
    assert dataset == "agent_sessions"
    assert kwargs["session_ids"] == ["chat_1", "chat_2"]
    assert kwargs["run_in_background"] is True
    assert kwargs["node_name"] == ["entity"]


def test_improve_warns_on_local_only_kwargs(stub_client, monkeypatch):
    warnings = []
    monkeypatch.setattr(state.logger, "warning", lambda *args, **kwargs: warnings.append(args))
    asyncio.run(cognee.improve("ds", session_ids=["s"], feedback_alpha=0.9))
    assert warnings and "feedback_alpha" in warnings[0][2]
    # The unsupported kwarg must not leak into the HTTP call either.
    _, _, kwargs = last_call(stub_client, "improve")
    assert "feedback_alpha" not in kwargs


def test_add_routes_node_set_and_dataset_id(stub_client):
    dataset_id = UUID("00000000-0000-0000-0000-000000000042")
    asyncio.run(
        cognee.add(
            "some text",
            "my_dataset",
            node_set=["user_context"],
            dataset_id=dataset_id,
            run_in_background=True,
        )
    )
    _, data, dataset_name, kwargs = last_call(stub_client, "add")
    assert data == "some text"
    assert dataset_name == "my_dataset"
    assert kwargs["node_set"] == ["user_context"]
    assert kwargs["dataset_id"] == dataset_id
    assert kwargs["run_in_background"] is True


def test_recall_default_auto_route_sends_explicit_null(stub_client):
    asyncio.run(cognee.recall("what do we know?"))
    _, _, query_type, _ = last_call(stub_client, "recall")
    assert query_type is None


def test_recall_without_auto_route_omits_search_type(stub_client):
    asyncio.run(cognee.recall("what do we know?", auto_route=False))
    _, _, query_type, _ = last_call(stub_client, "recall")
    assert query_type is UNSET


def test_recall_explicit_query_type_is_pinned(stub_client):
    asyncio.run(cognee.recall("what do we know?", query_type=SearchType.CHUNKS))
    _, _, query_type, _ = last_call(stub_client, "recall")
    assert query_type is SearchType.CHUNKS


def test_remember_routes_node_set_and_session_id(stub_client):
    asyncio.run(
        cognee.remember(
            "a note",
            dataset_name="agent_sessions",
            session_id="oc_session",
            node_set=["qa"],
        )
    )
    _, data, dataset_name, kwargs = last_call(stub_client, "remember")
    assert data == "a note"
    assert dataset_name == "agent_sessions"
    assert kwargs["node_set"] == ["qa"]
    assert kwargs["session_id"] == "oc_session"


def test_remember_warns_on_local_only_kwargs(stub_client, monkeypatch):
    warnings = []
    monkeypatch.setattr(state.logger, "warning", lambda *args, **kwargs: warnings.append(args))
    asyncio.run(cognee.remember("a note", preferred_loaders=["pdf"]))
    assert warnings and "preferred_loaders" in warnings[0][2]
    _, _, _, kwargs = last_call(stub_client, "remember")
    assert "preferred_loaders" not in kwargs


def test_search_forwards_session_id(stub_client, monkeypatch):
    # The server's search DTO accepts session_id (history + guidance feed
    # the completion), so it must reach the client rather than be warned away.
    warnings = []
    monkeypatch.setattr(state.logger, "warning", lambda *args, **kwargs: warnings.append(args))
    asyncio.run(cognee.search("query", session_id="oc_session"))
    assert not warnings
    _, _, kwargs = last_call(stub_client, "search")
    assert kwargs["session_id"] == "oc_session"
