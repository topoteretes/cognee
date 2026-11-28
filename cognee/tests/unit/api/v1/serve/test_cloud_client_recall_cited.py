"""Cited and expand calls over serve() go only to a server that lists recall_cited_v1.

Request bodies ignore unknown fields, so a server older than ``resultMode`` and
``expandItems`` answers a cited call with an untagged prompt and an expand call with an
ordinary recall of the query: no error, results the caller did not ask for. The
client reads the ``capabilities`` the server's ``/health`` lists (cached per client)
and refuses the request instead. A plain recall never probes.
"""

import asyncio
from uuid import uuid4

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.modules.recall.types.RecallCited import ExpandItem


class _Response:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    async def text(self):
        return str(self._payload)


class _Stalled(_Response):
    async def __aenter__(self):
        raise asyncio.TimeoutError


class _FakeServer:
    """A server's ``/health`` (None: not JSON, "stall": times out) and what it got."""

    def __init__(self, health, status=200):
        self.health = health
        self.status = status
        self.health_reads = 0
        self.posts = []

    def get(self, url, timeout=None):
        assert url.endswith("/health")
        self.health_reads += 1
        if self.health == "stall":
            return _Stalled(None, None)
        if self.health is None:
            return _Response(self.status, ValueError("not JSON"))
        return _Response(self.status, self.health)

    def post(self, url, *, json):
        self.posts.append(json)
        return _Response(200, [])


def _client(health, status=200):
    client = CloudClient("http://example.test", "key")
    server = _FakeServer(health, status)

    async def session():
        return server

    client._get_session = session
    return client, server


CURRENT = {"status": "ready", "capabilities": ["session_summary_v1", "recall_cited_v1"]}
ITEM = ExpandItem(dataset_id=uuid4(), record_kind="passage", record_id=uuid4())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs", [{"result_mode": "cited", "only_context": True}, {"expand_items": [ITEM]}]
)
async def test_a_server_that_lists_the_capability_gets_the_fields(kwargs):
    client, server = _client(CURRENT)
    await client.recall("q", **kwargs)
    await client.recall("q", **kwargs)

    assert server.health_reads == 1
    body = server.posts[-1]
    if "expand_items" in kwargs:
        assert body["expand_items"] == [ITEM.model_dump(mode="json")]
    else:
        assert body["result_mode"] == "cited"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "health, status",
    [
        ({"status": "ready", "capabilities": ["session_summary_v1"]}, 200),
        # The Cloud pod's /health before it listed capabilities: the report alone.
        ({"status": "healthy", "components": {}}, 200),
    ],
)
@pytest.mark.parametrize(
    "kwargs", [{"result_mode": "cited", "only_context": True}, {"expand_items": [ITEM]}]
)
async def test_an_older_server_is_refused_before_anything_is_sent(health, status, kwargs):
    client, server = _client(health, status)
    with pytest.raises(RuntimeError, match="does not list recall_cited_v1"):
        await client.recall("q", **kwargs)
    assert server.posts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("health", [None, "stall"])
async def test_an_unreadable_health_is_refused_and_asked_again(health):
    client, server = _client(health)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="is unknown"):
            await client.recall("q", result_mode="cited", only_context=True)
    assert server.posts == []
    assert server.health_reads == 2


@pytest.mark.asyncio
async def test_a_degraded_server_still_says_what_it_supports():
    client, server = _client({**CURRENT, "status": "not ready"}, status=503)
    await client.recall("q", result_mode="cited", only_context=True)
    assert server.posts[-1]["result_mode"] == "cited"


@pytest.mark.asyncio
async def test_an_error_body_without_the_list_is_unknown_not_unsupported():
    # cognee's own /health when its check fails, or a proxy's JSON error page: not an
    # older server, so nothing is cached and the next call asks again.
    client, server = _client({"status": "not ready", "reason": "health check failed"}, status=503)
    with pytest.raises(RuntimeError, match="is unknown"):
        await client.recall("q", result_mode="cited", only_context=True)
    server.health, server.status = CURRENT, 200
    await client.recall("q", result_mode="cited", only_context=True)
    assert server.health_reads == 2
    assert server.posts[-1]["result_mode"] == "cited"


@pytest.mark.asyncio
async def test_a_plain_recall_never_probes():
    client, server = _client(None)
    await client.recall("q")
    assert server.health_reads == 0
    assert "result_mode" not in server.posts[0] and "expand_items" not in server.posts[0]
