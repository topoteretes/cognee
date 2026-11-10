"""A typed node_type over serve() must never be sent to a server that cannot read it.

Request bodies ignore unknown fields, so a server older than ``nodeType`` accepts
the body and reads ``nodeName`` as node-set tags: no error, results for the wrong
scope. The client probes ``/openapi.json`` once and refuses the request instead,
the way ``find_data()`` refuses a content-hash lookup. The node-set default needs
no probe: every server reads it.
"""

from unittest.mock import AsyncMock

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.modules.engine.models.node_set import NodeSet
from cognee.tasks.schema.models import SchemaColumn


def _openapi(*, with_node_type: bool) -> dict:
    properties = {"query": {}, "nodeName": {}}
    if with_node_type:
        properties["nodeType"] = {}
    return {
        "components": {
            "schemas": {
                "SearchPayloadDTO": {"properties": dict(properties)},
                "RecallPayloadDTO": {"properties": dict(properties)},
            }
        }
    }


class _Response:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _FakeServer:
    """One generation of server: its schema (None: no schema served) and what it got."""

    def __init__(self, openapi):
        self.openapi = openapi
        self.schema_reads = 0
        self.posts = []

    def get(self, url):
        assert url.endswith("/openapi.json")
        self.schema_reads += 1
        if self.openapi is None:
            return _Response(404, {"detail": "Not Found"})
        return _Response(200, self.openapi)

    def post(self, url, *, json):
        self.posts.append((url.rsplit("/", 1)[-1], json))
        return _Response(200, [])


def _client(openapi):
    server = _FakeServer(openapi)
    client = CloudClient("https://example.test", "key")
    client._get_session = AsyncMock(return_value=server)
    return client, server


@pytest.mark.asyncio
async def test_current_server_gets_the_typed_scope_and_is_probed_once():
    client, server = _client(_openapi(with_node_type=True))

    await client.search("q", node_type=SchemaColumn, node_name=["genre:name"])
    await client.recall("q", None, node_type=SchemaColumn, node_name=["genre:name"])

    assert [
        (route, body.get("nodeType", body.get("node_type"))) for route, body in server.posts
    ] == [
        ("search", "SchemaColumn"),
        ("recall", "SchemaColumn"),
    ]
    assert server.schema_reads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("call", ["search", "recall"])
async def test_old_server_refuses_the_typed_scope_before_sending(call):
    client, server = _client(_openapi(with_node_type=False))

    with pytest.raises(RuntimeError, match="predates the field"):
        if call == "search":
            await client.search("q", node_type=SchemaColumn, node_name=["genre:name"])
        else:
            await client.recall("q", None, node_type=SchemaColumn, node_name=["genre:name"])

    assert server.posts == []
    assert client._node_type_supported is False


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", [NodeSet, None])
async def test_old_server_still_takes_the_node_set_default_without_a_probe(node_type):
    client, server = _client(_openapi(with_node_type=False))

    await client.search("q", node_type=node_type, node_name=["proj"])
    await client.recall("q", None, node_type=node_type, node_name=["proj"])

    assert [body["nodeName"] for route, body in server.posts if route == "search"] == [["proj"]]
    assert [body["node_name"] for route, body in server.posts if route == "recall"] == [["proj"]]
    assert all("nodeType" not in body and "node_type" not in body for _, body in server.posts)
    assert server.schema_reads == 0


@pytest.mark.asyncio
async def test_unreadable_schema_sends_the_request_and_asks_again_next_time():
    client, server = _client(None)

    await client.search("q", node_type=SchemaColumn, node_name=["genre:name"])
    await client.search("q", node_type=SchemaColumn, node_name=["genre:name"])

    assert [body["nodeType"] for _, body in server.posts] == ["SchemaColumn", "SchemaColumn"]
    assert server.schema_reads == 2
    assert client._node_type_supported is None
