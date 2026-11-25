"""CloudClient must put node_set, labels, external_metadata and
retriever_specific_config on the wire.

The server endpoints accept all four, but the client silently dropped them, so
an SDK example switched to CloudClient lost its node-set tags, per-file labels
and metadata, and every retriever option (issues #6706, #6692). These tests pin
the wire layer: the exact form fields and JSON payload keys the client POSTs.
"""

import json
from unittest.mock import AsyncMock

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient


class _FakeResponse:
    status = 200

    async def json(self):
        return {}

    async def text(self):
        return ""


class _FakePost:
    def __init__(self, recorder, url, kwargs):
        recorder.append((url, kwargs))

    async def __aenter__(self):
        return _FakeResponse()

    async def __aexit__(self, *exc):
        return False


def _client_with_recorder():
    client = CloudClient.__new__(CloudClient)
    client.service_url = "http://server"
    posts = []

    class FakeSession:
        def post(self, url, **kwargs):
            return _FakePost(posts, url, kwargs)

    client._get_session = AsyncMock(return_value=FakeSession())
    return client, posts


def _form_fields(form):
    """(name, value) pairs of a multipart form, bytes decoded."""
    fields = []
    for disposition, _headers, value in form._fields:
        if isinstance(value, bytes):
            value = value.decode()
        fields.append((disposition["name"], value))
    return fields


@pytest.mark.asyncio
async def test_remember_forwards_node_set_labels_and_external_metadata():
    client, posts = _client_with_recorder()

    await client.remember(
        "some text",
        node_set=["team-a", "project-x"],
        labels=["finance", "people"],
        external_metadata=[{"source": "crm", "ticket": 42}],
    )

    url, kwargs = posts[0]
    assert url == "http://server/api/v1/remember"
    fields = _form_fields(kwargs["data"])
    assert fields.count(("node_set", "team-a")) == 1
    assert fields.count(("node_set", "project-x")) == 1
    # labels/external_metadata travel as one JSON array string each, paired
    # positionally — the encoding the server's parse_labels/parse_external_metadata read.
    assert ("labels", json.dumps(["finance", "people"])) in fields
    assert ("external_metadata", json.dumps([{"source": "crm", "ticket": 42}])) in fields


@pytest.mark.asyncio
async def test_remember_omits_the_fields_when_not_given():
    client, posts = _client_with_recorder()

    await client.remember("some text")

    fields = _form_fields(posts[0][1]["data"])
    names = {name for name, _ in fields}
    assert "node_set" not in names
    assert "labels" not in names
    assert "external_metadata" not in names


@pytest.mark.asyncio
async def test_add_forwards_node_set_labels_and_external_metadata():
    client, posts = _client_with_recorder()

    await client.add(
        "some text",
        node_set=["team-a"],
        labels=["finance"],
        external_metadata=[{"source": "crm"}],
    )

    url, kwargs = posts[0]
    assert url == "http://server/api/v1/add"
    fields = _form_fields(kwargs["data"])
    assert ("node_set", "team-a") in fields
    assert ("labels", json.dumps(["finance"])) in fields
    assert ("external_metadata", json.dumps([{"source": "crm"}])) in fields


@pytest.mark.asyncio
async def test_recall_forwards_retriever_specific_config():
    client, posts = _client_with_recorder()

    await client.recall(
        "what is in the doc?",
        retriever_specific_config={"chunks_top_k": 3, "include_external_metadata": True},
    )

    url, kwargs = posts[0]
    assert url == "http://server/api/v1/recall"
    assert kwargs["json"]["retriever_specific_config"] == {
        "chunks_top_k": 3,
        "include_external_metadata": True,
    }


@pytest.mark.asyncio
async def test_search_forwards_retriever_specific_config():
    client, posts = _client_with_recorder()

    await client.search("what is in the doc?", retriever_specific_config={"chunks_top_k": 3})

    url, kwargs = posts[0]
    assert url == "http://server/api/v1/search"
    assert kwargs["json"]["retriever_specific_config"] == {"chunks_top_k": 3}
