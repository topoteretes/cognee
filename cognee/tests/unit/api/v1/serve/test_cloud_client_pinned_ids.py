"""After ``serve()``, a ``DataItem``'s pinned ``data_id`` must reach the server.

``CloudClient.add()``/``remember()`` used to drop ``DataItem`` wrappers on the
floor (neither a string nor file-like), so the id, label and metadata never
left the SDK and the server minted its own UUID — callers had no stable
handle for a later ``update()``. They now unwrap the payload and send the
attributes as the positional JSON array fields the routes accept
(``labels`` / ``external_metadata`` / ``data_ids``), plus ``node_set`` and
``datasetId``, which were dropped as well. A server too old to know
``data_ids`` ignores the field silently, so the client checks the ids the
response reports against the pins and fails loudly on a mismatch.
"""

import io
import json
import logging
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient, _verify_pinned_ids
from cognee.tasks.ingestion.data_item import DataItem


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _Session:
    def __init__(self, payload):
        self.calls = []
        self._payload = payload

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self._payload)


def _fields(form):
    return [(opts["name"], opts.get("filename"), value) for opts, _h, value in form._fields]


def _form_values(form, name):
    return [value for field, _, value in _fields(form) if field == name]


@pytest.mark.asyncio
async def test_add_sends_pinned_ids_labels_metadata_node_set_and_dataset_id():
    pinned, dataset_id = uuid4(), uuid4()
    client = CloudClient("https://example.test", "key")
    session = _Session(
        {
            "status": "PipelineRunCompleted",
            "data_ingestion_info": [{"data_id": str(pinned)}, {"data_id": str(uuid4())}],
        }
    )
    client._get_session = AsyncMock(return_value=session)

    await client.add(
        [
            DataItem(
                data="pinned text", label="finance", external_metadata={"k": 1}, data_id=pinned
            ),
            "plain",
        ],
        "repro",
        dataset_id=dataset_id,
        node_set=["a", "b"],
    )

    ((url, kwargs),) = session.calls
    assert url == "https://example.test/api/v1/add"
    form = kwargs["data"]
    assert _form_values(form, "datasetName") == ["repro"]
    assert _form_values(form, "datasetId") == [str(dataset_id)]
    assert _form_values(form, "node_set") == ["a", "b"]
    assert json.loads(_form_values(form, "data_ids")[0]) == [str(pinned), None]
    assert json.loads(_form_values(form, "labels")[0]) == ["finance", None]
    assert json.loads(_form_values(form, "external_metadata")[0]) == [{"k": 1}, None]
    uploads = [(fn, v.read()) for name, fn, v in _fields(form) if name == "data"]
    assert [body for _, body in uploads] == [b"pinned text", b"plain"]


@pytest.mark.asyncio
async def test_remember_sends_pinned_ids_and_node_set():
    pinned = uuid4()
    client = CloudClient("https://example.test", "key")
    session = _Session({"status": "completed", "items": [{"id": str(pinned)}]})
    client._get_session = AsyncMock(return_value=session)

    await client.remember(DataItem(data="hello", data_id=pinned), "repro", node_set=["repro"])

    ((url, kwargs),) = session.calls
    assert url == "https://example.test/api/v1/remember"
    form = kwargs["data"]
    assert json.loads(_form_values(form, "data_ids")[0]) == [str(pinned)]
    assert _form_values(form, "node_set") == ["repro"]
    assert "labels" not in [name for name, _, _ in _fields(form)]


@pytest.mark.asyncio
async def test_unpinned_uploads_send_no_attribute_fields():
    client = CloudClient("https://example.test", "key")
    session = _Session({"status": "completed"})
    client._get_session = AsyncMock(return_value=session)
    handle = io.BytesIO(b"bytes")
    handle.name = "/somewhere/notes.txt"

    await client.add(["text", handle], "repro")

    ((_, kwargs),) = session.calls
    names = [name for name, _, _ in _fields(kwargs["data"])]
    assert names == ["datasetName", "data", "data"]
    assert [fn for name, fn, _ in _fields(kwargs["data"]) if name == "data"][1] == "notes.txt"


@pytest.mark.asyncio
async def test_add_uploads_local_file_bytes_for_path_strings(tmp_path):
    source = tmp_path / "doc.md"
    source.write_text("on disk")
    client = CloudClient("https://example.test", "key")
    session = _Session({"status": "completed"})
    client._get_session = AsyncMock(return_value=session)

    await client.add(str(source), "repro")

    ((_, kwargs),) = session.calls
    ((_, filename, value),) = [f for f in _fields(kwargs["data"]) if f[0] == "data"]
    assert filename == "doc.md"
    assert value.read() == b"on disk"
    value.close()


@pytest.mark.asyncio
async def test_add_raises_when_server_ignores_the_pin():
    pinned, minted = uuid4(), uuid4()
    client = CloudClient("https://example.test", "key")
    client._get_session = AsyncMock(
        return_value=_Session({"data_ingestion_info": [{"data_id": str(minted)}]})
    )

    with pytest.raises(RuntimeError, match=f"did not honor pinned data_id.*{pinned}"):
        await client.add(DataItem(data="x", data_id=pinned), "repro")


def test_verify_pinned_ids_warns_when_response_reports_no_ids(caplog):
    pinned = str(uuid4())
    with caplog.at_level(logging.WARNING):
        _verify_pinned_ids({"status": "PipelineRunStarted"}, [pinned], "add")
    assert any(pinned in record.getMessage() for record in caplog.records)

    # Nothing pinned → nothing to verify, whatever the response says.
    _verify_pinned_ids({}, [None, None], "add")
    _verify_pinned_ids("not a dict", [], "remember")
