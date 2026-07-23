"""Remote (serve) mode must forward improve()'s session parameters.

Before the fix, ``improve(session_ids=[...])`` against a remote backend
silently dropped ``session_ids``, ``run_in_background`` and
``build_global_context_index``: they are keyword-only parameters, so they
never reached ``client.improve(dataset, node_name=..., **kwargs)``, and the
CloudClient did not serialize them either. The call returned 200 but only ran
the default enrichment stage — no session bridging.
"""

from importlib import import_module
from unittest.mock import AsyncMock, patch

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient


class DummySpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def set_attribute(self, key, value):
        pass


@pytest.mark.asyncio
async def test_improve_remote_dispatch_forwards_session_params(monkeypatch):
    import cognee.shared.utils as shared_utils

    improve_module = import_module("cognee.api.v1.improve.improve")
    serve_state = import_module("cognee.api.v1.serve.state")

    remote_client = AsyncMock()
    remote_client.improve = AsyncMock(return_value={"status": "completed"})

    monkeypatch.setattr(shared_utils, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: remote_client)
    monkeypatch.setattr(improve_module, "new_span", lambda _: DummySpan())

    result = await improve_module.improve(
        dataset="docs",
        session_ids=["chat_1", "chat_2"],
        run_in_background=True,
        build_global_context_index=True,
        node_name=["entity"],
    )

    assert result == {"status": "completed"}
    remote_client.improve.assert_awaited_once()
    args, kwargs = remote_client.improve.await_args
    assert args == ("docs",)
    assert kwargs["session_ids"] == ["chat_1", "chat_2"]
    assert kwargs["run_in_background"] is True
    assert kwargs["build_global_context_index"] is True
    assert kwargs["node_name"] == ["entity"]


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return {"status": "completed"}

    async def text(self):
        return ""


class _FakeSession:
    def __init__(self):
        self.last_url = None
        self.last_json = None

    def post(self, url, json=None, **kwargs):
        self.last_url = url
        self.last_json = json
        return _FakeResponse()


@pytest.mark.asyncio
async def test_cloud_client_improve_serializes_session_params():
    client = CloudClient(service_url="https://tenant.example.com", api_key="key")
    fake_session = _FakeSession()

    with patch.object(client, "_get_session", new=AsyncMock(return_value=fake_session)):
        result = await client.improve(
            "docs",
            session_ids=["chat_1"],
            run_in_background=True,
            build_global_context_index=True,
            node_name=["entity"],
        )

    assert result == {"status": "completed"}
    assert fake_session.last_url == "https://tenant.example.com/api/v1/improve"
    assert fake_session.last_json == {
        "dataset_name": "docs",
        "run_in_background": True,
        "node_name": ["entity"],
        "session_ids": ["chat_1"],
        "build_global_context_index": True,
    }


@pytest.mark.asyncio
async def test_cloud_client_improve_omits_unset_session_params():
    client = CloudClient(service_url="https://tenant.example.com", api_key="key")
    fake_session = _FakeSession()

    with patch.object(client, "_get_session", new=AsyncMock(return_value=fake_session)):
        await client.improve("docs")

    assert fake_session.last_json == {"dataset_name": "docs"}
