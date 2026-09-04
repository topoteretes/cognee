"""update() must run on the remote instance once cognee.serve() is connected.

Regression tests for the report that update() after serve() ran against the
LOCAL store: it resolved the local default user and deleted locally against
the remote dataset id, failing with "Dataset not found" while the remote
document stayed untouched."""

import importlib
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from cognee.api.v1.serve import state as state_mod
from cognee.api.v1.serve.cloud_client import CloudClient

# ``cognee.api.v1.update`` re-exports the update() *function* under the same
# name as its module, so a plain ``from ... import update`` yields the
# function; resolve the module explicitly to patch its globals.
update_mod = importlib.import_module("cognee.api.v1.update.update")


class _StubClient:
    def __init__(self):
        self.calls = []

    async def update(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "incremental", "data_id": str(kwargs["data_id"])}


@pytest.fixture
def remote_stub(monkeypatch):
    stub = _StubClient()
    monkeypatch.setattr(state_mod, "_remote_client", stub)

    async def local_path_reached():
        raise AssertionError("update() did local work while a remote client was connected")

    monkeypatch.setattr(update_mod, "get_default_user", local_path_reached)
    return stub


@pytest.mark.asyncio
async def test_update_routes_to_remote_before_any_local_work(remote_stub):
    data_id, dataset_id = uuid4(), uuid4()

    result = await update_mod.update(
        data_id=data_id,
        data="new text",
        dataset_id=dataset_id,
        node_set=["serve"],
        chunk_level_diff=False,
    )

    assert result["status"] == "incremental"
    assert remote_stub.calls == [
        {
            "data_id": data_id,
            "data": "new text",
            "dataset_id": dataset_id,
            "node_set": ["serve"],
            "chunk_level_diff": False,
        }
    ]


@pytest.mark.asyncio
async def test_update_remote_warns_about_parameters_the_route_cannot_carry(
    remote_stub, monkeypatch
):
    mock_logger = MagicMock()
    monkeypatch.setattr(update_mod, "logger", mock_logger)

    await update_mod.update(
        data_id=uuid4(), data="new text", dataset_id=uuid4(), custom_prompt="be brief"
    )

    assert mock_logger.warning.call_count == 1
    assert "custom_prompt" in mock_logger.warning.call_args.args[1]


@pytest.mark.asyncio
async def test_update_remote_is_silent_when_only_routable_parameters_are_given(
    remote_stub, monkeypatch
):
    mock_logger = MagicMock()
    monkeypatch.setattr(update_mod, "logger", mock_logger)

    await update_mod.update(data_id=uuid4(), data="new text", dataset_id=uuid4())

    mock_logger.warning.assert_not_called()


# ----- CloudClient.update wire format -----


class _FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload or {}
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


def _client_with_fake_patch(monkeypatch, response):
    client = CloudClient("http://remote.invalid", "key")
    captured = {}

    @asynccontextmanager
    async def fake_patch(url, params=None, data=None):
        captured.update(url=url, params=params, form=data)
        yield response

    session = MagicMock()
    session.patch = fake_patch

    async def get_session():
        return session

    monkeypatch.setattr(client, "_get_session", get_session)
    return client, captured


def _field_names(form):
    return [options["name"] for options, _headers, _value in form._fields]


@pytest.mark.asyncio
async def test_cloud_client_update_matches_the_route_contract(monkeypatch):
    client, captured = _client_with_fake_patch(monkeypatch, _FakeResponse(payload={"ok": 1}))
    data_id, dataset_id = uuid4(), uuid4()

    result = await client.update(
        data_id=data_id,
        data="new text",
        dataset_id=dataset_id,
        node_set=["a", "b"],
        chunk_level_diff=False,
    )

    assert result == {"ok": 1}
    assert captured["url"] == "http://remote.invalid/api/v1/update"
    assert captured["params"] == {
        "data_id": str(data_id),
        "dataset_id": str(dataset_id),
        "chunk_level_diff": "false",
    }
    assert _field_names(captured["form"]) == ["data", "node_set", "node_set"]


@pytest.mark.asyncio
async def test_cloud_client_update_unwraps_single_item_list_and_data_item(monkeypatch):
    from cognee.tasks.ingestion.data_item import DataItem

    client, captured = _client_with_fake_patch(monkeypatch, _FakeResponse())

    await client.update(data_id=uuid4(), data=[DataItem(data="wrapped")], dataset_id=uuid4())

    assert _field_names(captured["form"]) == ["data"]


@pytest.mark.asyncio
async def test_cloud_client_update_rejects_multiple_documents(monkeypatch):
    client, _ = _client_with_fake_patch(monkeypatch, _FakeResponse())

    with pytest.raises(ValueError, match="exactly one document"):
        await client.update(data_id=uuid4(), data=["one", "two"], dataset_id=uuid4())


@pytest.mark.asyncio
async def test_cloud_client_update_surfaces_remote_errors(monkeypatch):
    client, _ = _client_with_fake_patch(
        monkeypatch, _FakeResponse(status=404, text='{"error":"not found"}')
    )

    with pytest.raises(RuntimeError, match=r"Remote update failed \(404\)"):
        await client.update(data_id=uuid4(), data="new text", dataset_id=uuid4())
