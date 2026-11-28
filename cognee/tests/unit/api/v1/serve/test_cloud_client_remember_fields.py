"""CloudClient.remember must forward node_set, labels and external_metadata (#6706).

Local remember() takes node_set as a kwarg and per-file labels and
external_metadata on DataItem wrappers. Over serve() the client dropped all
three, so the remote stored the data untagged. These tests pin the multipart
form the client POSTs to /api/v1/remember.
"""

import json
from unittest.mock import AsyncMock

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.tasks.ingestion.data_item import DataItem


class _FakeResponse:
    status = 200

    async def json(self):
        return {"status": "ok"}

    async def text(self):
        return ""


class _FakePost:
    def __init__(self, recorder, url, data, timeout):
        recorder.append((url, data))

    async def __aenter__(self):
        return _FakeResponse()

    async def __aexit__(self, *exc):
        return False


def _client_with_recorder():
    client = CloudClient.__new__(CloudClient)
    client.service_url = "http://server"
    posts = []
    session = type(
        "FakeSession",
        (),
        {"post": lambda self, url, data, timeout: _FakePost(posts, url, data, timeout)},
    )()
    client._get_session = AsyncMock(return_value=session)
    return client, posts


def _fields(form) -> list[tuple[str, object]]:
    return [(options["name"], value) for options, _headers, value in form._fields]


def _values(form, name: str) -> list:
    return [value for field_name, value in _fields(form) if field_name == name]


@pytest.mark.asyncio
async def test_node_set_is_sent_as_repeated_form_fields():
    client, posts = _client_with_recorder()

    await client.remember("some text", "docs", node_set=["finance", "", "q3"])

    url, form = posts[0]
    assert url == "http://server/api/v1/remember"
    assert _values(form, "node_set") == ["finance", "q3"]


@pytest.mark.asyncio
async def test_data_item_labels_and_metadata_pair_with_uploads_by_position():
    client, posts = _client_with_recorder()

    await client.remember(
        [
            DataItem("first doc", label="finance", external_metadata={"source": "crm"}),
            "plain text",
            DataItem("third doc", label="people"),
        ],
        "docs",
    )

    _url, form = posts[0]
    assert len(_values(form, "data")) == 3
    assert json.loads(_values(form, "labels")[0]) == ["finance", None, "people"]
    assert json.loads(_values(form, "external_metadata")[0]) == [{"source": "crm"}, None, None]


@pytest.mark.asyncio
async def test_single_data_item_is_unwrapped_and_uploaded():
    client, posts = _client_with_recorder()

    await client.remember(DataItem("only doc", label="notes"), "docs")

    _url, form = posts[0]
    (upload,) = _values(form, "data")
    assert upload.read() == b"only doc"
    assert json.loads(_values(form, "labels")[0]) == ["notes"]
    assert _values(form, "external_metadata") == []


@pytest.mark.asyncio
async def test_plain_data_sends_no_labels_or_metadata():
    client, posts = _client_with_recorder()

    await client.remember(["a", "b"], "docs")

    _url, form = posts[0]
    assert _values(form, "labels") == []
    assert _values(form, "external_metadata") == []
    assert _values(form, "node_set") == []


@pytest.mark.asyncio
async def test_per_item_node_set_raises_instead_of_being_dropped():
    client, posts = _client_with_recorder()

    with pytest.raises(ValueError, match="node_set"):
        await client.remember([DataItem("doc", node_set=["team-a"])], "docs")
    assert posts == []
