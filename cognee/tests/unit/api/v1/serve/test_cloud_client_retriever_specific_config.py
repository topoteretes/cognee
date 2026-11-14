"""retriever_specific_config must be forwarded by both recall() and search().

Both methods build a JSON payload from kwargs and POST it to the server.
Before this fix, retriever_specific_config was silently dropped, so callers
could not pass retriever-level options (chunks_top_k, entities_top_k, …)
through the cloud client. The server accepts and validates the field on both
/api/v1/recall and /api/v1/search.
"""

from unittest.mock import AsyncMock

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient


class _Response:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _FakeServer:
    """Records every POST body for inspection."""

    def __init__(self):
        self.posts = []

    def get(self, url, timeout=None):
        # Return a minimal OpenAPI schema so node_type probes don't block.
        return _Response(
            200,
            {
                "paths": {},
                "components": {
                    "schemas": {
                        "SearchPayloadDTO": {"properties": {"query": {}}},
                        "RecallPayloadDTO": {"properties": {"query": {}}},
                    }
                },
            },
        )

    def post(self, url, *, json):
        route = url.rsplit("/", 1)[-1]
        self.posts.append((route, json))
        return _Response(200, [])


def _client():
    server = _FakeServer()
    client = CloudClient("https://example.test", "key")
    client._get_session = AsyncMock(return_value=server)
    return client, server


@pytest.mark.asyncio
async def test_recall_forwards_retriever_specific_config():
    """recall() must include retriever_specific_config in the POST body."""
    client, server = _client()
    config = {"chunks_top_k": 5, "entities_top_k": 10}

    await client.recall("what is cognee?", retriever_specific_config=config)

    assert len(server.posts) == 1
    route, body = server.posts[0]
    assert route == "recall"
    assert body.get("retriever_specific_config") == config


@pytest.mark.asyncio
async def test_search_forwards_retriever_specific_config():
    """search() must include retriever_specific_config in the POST body."""
    client, server = _client()
    config = {"chunks_top_k": 20, "facts_top_k": 15}

    await client.search("what is cognee?", retriever_specific_config=config)

    assert len(server.posts) == 1
    route, body = server.posts[0]
    assert route == "search"
    assert body.get("retriever_specific_config") == config


@pytest.mark.asyncio
async def test_recall_omits_retriever_specific_config_when_none():
    """When retriever_specific_config is not passed, the key must not appear in the payload."""
    client, server = _client()

    await client.recall("what is cognee?")

    assert len(server.posts) == 1
    _, body = server.posts[0]
    assert "retriever_specific_config" not in body


@pytest.mark.asyncio
async def test_search_omits_retriever_specific_config_when_none():
    """When retriever_specific_config is not passed, the key must not appear in the payload."""
    client, server = _client()

    await client.search("what is cognee?")

    assert len(server.posts) == 1
    _, body = server.posts[0]
    assert "retriever_specific_config" not in body


@pytest.mark.asyncio
async def test_recall_forwards_retriever_specific_config_alongside_other_kwargs():
    """retriever_specific_config is forwarded even when other kwargs are also set."""
    client, server = _client()
    config = {"entities_top_k": 7}

    await client.recall(
        "what is cognee?",
        top_k=5,
        verbose=True,
        retriever_specific_config=config,
    )

    assert len(server.posts) == 1
    _, body = server.posts[0]
    assert body.get("retriever_specific_config") == config
    assert body.get("top_k") == 5
    assert body.get("verbose") is True

