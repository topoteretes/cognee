"""update() must run on the remote instance once cognee.serve() is connected.

Regression tests for the report that update() after serve() ran against the
LOCAL store: it resolved the local default user and deleted locally against
the remote dataset id, failing with "Dataset not found" while the remote
document stayed untouched."""

import importlib
import io
import json
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from cognee.api.v1.serve import state as state_mod
from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.api.v1.update import UpdateBatchResult, UpdateResult

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


def _result_payload(**overrides):
    """A PATCH /update body as the server sends it."""
    payload = {
        "status": "incremental",
        "regions": 1,
        "deleted_chunks": 1,
        "added_chunks": 1,
        "reused_chunks": 0,
        "kept_chunks": 3,
        "reindexed_chunks": 0,
        "total_chunks": 4,
        "data_id": str(uuid4()),
        "dataset_id": str(uuid4()),
        "duration_seconds": 0.8,
        "pipeline_run_id": str(uuid4()),
        "fallback": None,
        "error": None,
    }
    payload.update(overrides)
    return payload


class _FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload if payload is not None else _result_payload()
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
    payload = _result_payload()
    client, captured = _client_with_fake_patch(monkeypatch, _FakeResponse(payload=payload))
    data_id, dataset_id = uuid4(), uuid4()

    result = await client.update(
        data_id=data_id,
        data="new text",
        dataset_id=dataset_id,
        node_set=["a", "b"],
        chunk_level_diff=False,
    )

    assert result == UpdateResult.model_validate(payload).model_dump()
    assert (result["status"], result["kept_chunks"]) == ("incremental", 3)
    assert result["data_id"] == UUID(payload["data_id"]), "remote ids are UUIDs, as locally"
    assert captured["url"] == "http://remote.invalid/api/v1/update"
    assert captured["params"] == {
        "data_id": str(data_id),
        "dataset_id": str(dataset_id),
        "chunk_level_diff": "false",
    }
    assert _field_names(captured["form"]) == ["data", "node_set", "node_set"]


def _batch_payload(statuses):
    results = [
        _result_payload(
            status=s,
            **({} if s == "incremental" else dict.fromkeys(COUNTERS)),
            error={"error_class": "RuntimeError", "message": "boom"} if s == "failed" else None,
            fallback=None if s == "incremental" else {"reason": "no_baseline", "detail": "x"},
        )
        for s in statuses
    ]
    failed = statuses.count("failed")
    return {
        "status": "failed" if failed == len(statuses) else "partial" if failed else "completed",
        "total": len(statuses),
        "updated": len(statuses) - failed,
        "unchanged": 0,
        "failed": failed,
        "dataset_id": str(uuid4()),
        "duration_seconds": 1.5,
        "results": results,
    }


COUNTERS = (
    "regions",
    "deleted_chunks",
    "added_chunks",
    "reused_chunks",
    "kept_chunks",
    "reindexed_chunks",
    "total_chunks",
)


@pytest.mark.asyncio
async def test_cloud_client_update_single_data_item_id_travels_as_the_query_param(monkeypatch):
    from cognee.tasks.ingestion.data_item import DataItem

    client, captured = _client_with_fake_patch(monkeypatch, _FakeResponse())
    doc = uuid4()

    await client.update(data=DataItem(data="wrapped", data_id=doc), dataset_id=uuid4())

    assert _field_names(captured["form"]) == ["data"]
    assert captured["params"]["data_id"] == str(doc)


@pytest.mark.asyncio
async def test_cloud_client_update_without_data_id_sends_none_so_the_server_infers(monkeypatch):
    client, captured = _client_with_fake_patch(monkeypatch, _FakeResponse())

    await client.update(data="new text", dataset_id=uuid4())

    assert "data_id" not in captured["params"]


@pytest.mark.asyncio
async def test_cloud_client_update_sends_a_local_file_under_its_filename(monkeypatch, tmp_path):
    """The server infers the document by filename, as a local call does by path."""
    report = tmp_path / "report.txt"
    report.write_bytes(b"edited")
    client, captured = _client_with_fake_patch(monkeypatch, _FakeResponse())

    await client.update(data=str(report), dataset_id=uuid4())

    (options, _headers, value), *_ = captured["form"]._fields
    assert options["filename"] == "report.txt"
    assert not isinstance(value, io.BytesIO), "the file itself is sent, not its path as text"


@pytest.mark.asyncio
async def test_cloud_client_update_sends_a_list_as_a_batch_and_parses_the_batch(monkeypatch):
    payload = _batch_payload(["incremental", "failed"])
    client, captured = _client_with_fake_patch(monkeypatch, _FakeResponse(payload=payload))

    result = await client.update(data=["one", "two"], dataset_id=uuid4())

    assert _field_names(captured["form"]) == ["data", "data"]
    assert "data_id" not in captured["params"]
    assert result == UpdateBatchResult.model_validate(payload).model_dump()
    assert (result["status"], result["failed"]) == ("partial", 1)
    assert result["results"][1]["data_id"] == UUID(payload["results"][1]["data_id"])


@pytest.mark.asyncio
async def test_cloud_client_update_refuses_per_document_ids_in_a_batch(monkeypatch):
    from cognee.tasks.ingestion.data_item import DataItem

    client, _ = _client_with_fake_patch(monkeypatch, _FakeResponse())

    with pytest.raises(ValueError, match="per-document data_ids cannot travel"):
        await client.update(
            data=[DataItem(data="a", data_id=uuid4()), DataItem(data="b", data_id=uuid4())],
            dataset_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_cloud_client_update_returns_a_failed_batch_instead_of_raising(monkeypatch):
    payload = _batch_payload(["failed", "failed"])
    client, _ = _client_with_fake_patch(
        monkeypatch, _FakeResponse(status=500, text=json.dumps(payload))
    )

    result = await client.update(data=["one", "two"], dataset_id=uuid4())

    assert result == UpdateBatchResult.model_validate(payload).model_dump()
    assert (result["status"], result["failed"]) == ("failed", 2)


@pytest.mark.asyncio
async def test_cloud_client_update_surfaces_remote_errors(monkeypatch):
    client, _ = _client_with_fake_patch(
        monkeypatch, _FakeResponse(status=404, text='{"error":"not found"}')
    )

    with pytest.raises(RuntimeError, match=r"Remote update failed \(404\)"):
        await client.update(data_id=uuid4(), data="new text", dataset_id=uuid4())


@pytest.mark.asyncio
async def test_cloud_client_update_returns_a_failed_result_instead_of_raising(monkeypatch):
    """The route answers a failed rebuild with 500 and the result body; the
    client hands that result back so the caller can read the error and retry."""
    failed = _result_payload(
        status="failed",
        **dict.fromkeys(
            (
                "regions",
                "deleted_chunks",
                "added_chunks",
                "reused_chunks",
                "kept_chunks",
                "reindexed_chunks",
                "total_chunks",
            ),
        ),
        fallback={"reason": "disabled", "detail": "chunk_level_diff=False was requested"},
        error={"error_class": "RuntimeError", "message": "cognify failed"},
    )
    client, _ = _client_with_fake_patch(
        monkeypatch, _FakeResponse(status=500, text=json.dumps(failed))
    )

    result = await client.update(data_id=uuid4(), data="new text", dataset_id=uuid4())

    assert result == UpdateResult.model_validate(failed).model_dump()
    assert (result["status"], result["error"]["message"]) == ("failed", "cognify failed")
