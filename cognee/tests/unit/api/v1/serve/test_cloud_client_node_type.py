"""A typed node_type over serve() must never be sent to a server that cannot read it.

Request bodies ignore unknown fields, so a server older than ``nodeType`` accepts
the body and reads ``nodeName`` as node-set tags: no error, results for the wrong
scope. The client probes ``/openapi.json`` once and refuses the request instead,
the way ``find_data()`` refuses a content-hash lookup. Both probes read one cached
schema fetch. No probe runs when the field has no effect: the node-set default, and
a typed class with no names, go to every server.
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
        "paths": {
            "/api/v1/datasets/{dataset_id}/data": {
                "get": {"parameters": [{"name": "content_hash", "in": "query"}]}
            }
        },
        "components": {
            "schemas": {
                "SearchPayloadDTO": {"properties": dict(properties)},
                "RecallPayloadDTO": {"properties": dict(properties)},
            }
        },
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
        self.schema_timeouts = []
        self.posts = []

    def get(self, url, timeout=None):
        assert url.endswith("/openapi.json")
        self.schema_reads += 1
        self.schema_timeouts.append(timeout)
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
    assert server.schema_timeouts == [CloudClient.SCHEMA_PROBE_TIMEOUT]


@pytest.mark.asyncio
async def test_both_feature_probes_read_one_schema_fetch():
    client, server = _client(_openapi(with_node_type=True))

    assert await client.supports_content_hash_lookup() is True
    assert await client.supports_node_type() is True
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
@pytest.mark.parametrize("openapi", [_openapi(with_node_type=False), None])
async def test_a_typed_class_without_names_goes_to_any_server_unprobed(openapi):
    """Without node_name the type changes nothing, so an old server answers correctly."""
    client, server = _client(openapi)

    await client.search("q", node_type=SchemaColumn)
    await client.recall("q", None, node_type=SchemaColumn)

    assert [body.get("nodeType", body.get("node_type")) for _, body in server.posts] == [
        "SchemaColumn",
        "SchemaColumn",
    ]
    assert server.schema_reads == 0


@pytest.mark.asyncio
async def test_unreadable_schema_refuses_the_typed_scope_and_asks_again_next_time():
    """Nothing in a response shows whether the server applied the type, so unknown
    fails closed; the answer is not cached, so a later call can still find out."""
    client, server = _client(None)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="Could not read the remote server's schema"):
            await client.search("q", node_type=SchemaColumn, node_name=["genre:name"])

    assert server.posts == []
    assert server.schema_reads == 2
    assert client._node_type_supported is None
