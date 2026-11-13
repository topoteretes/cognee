"""CloudClient.add must unwrap DataItem and forward dataset_id/node_set/run_in_background."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.tasks.ingestion.data_item import DataItem


class _FakeResponse:
    status = 200

    async def json(self):
        return {"ok": True}

    async def text(self):
        return ""


class _FakePost:
    def __init__(self, recorder, url, data):
        recorder.append((url, data))

    async def __aenter__(self):
        return _FakeResponse()

    async def __aexit__(self, *exc):
        return False


def _client_with_form_recorder():
    client = CloudClient.__new__(CloudClient)
    client.service_url = "http://server"
    posts = []
    session = type(
        "FakeSession",
        (),
        {"post": lambda self, url, data: _FakePost(posts, url, data)},
    )()
    client._get_session = AsyncMock(return_value=session)
    return client, posts


def _form_fields(form):
    # aiohttp FormData stores (MultiDict headers, attrs, value)
    fields = []
    for headers, _attrs, value in form._fields:
        name = headers["name"]
        if hasattr(value, "getvalue"):
            value = value.getvalue().decode("utf-8")
        elif hasattr(value, "read"):
            raw = value.read()
            if hasattr(value, "seek"):
                value.seek(0)
            value = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        fields.append((name, value))
    return fields


@pytest.mark.asyncio
async def test_add_unwraps_data_item():
    client, posts = _client_with_form_recorder()
    await client.add(DataItem(data="hello world", label="greeting"), "main_dataset")
    url, form = posts[0]
    assert url == "http://server/api/v1/add"
    fields = _form_fields(form)
    names = [n for n, _ in fields]
    assert "datasetName" in names
    assert any(v == "hello world" for _, v in fields)


@pytest.mark.asyncio
async def test_add_unwraps_data_item_in_list():
    client, posts = _client_with_form_recorder()
    await client.add(["plain", DataItem(data="wrapped")], "main_dataset")
    _, form = posts[0]
    values = [v for _, v in _form_fields(form) if _ != "datasetName"]
    # both contents should be present
    assert "plain" in values
    assert "wrapped" in values


@pytest.mark.asyncio
async def test_add_forwards_dataset_id_node_set_and_background():
    client, posts = _client_with_form_recorder()
    dataset_id = uuid4()
    await client.add(
        "hello",
        "ignored_name",
        dataset_id=dataset_id,
        node_set=["team-a", "team-b"],
        run_in_background=True,
    )
    _, form = posts[0]
    fields = dict(_form_fields(form))
    assert fields.get("datasetId") == str(dataset_id)
    assert fields.get("datasetName") == "ignored_name"
    assert fields.get("run_in_background") == "true"
    node_set_values = [v for n, v in _form_fields(form) if n == "node_set"]
    assert node_set_values == ["team-a", "team-b"]
