"""A hung embedding attempt leaves time for another request (CLO-1869)."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tenacity import wait_none

from cognee.infrastructure.databases.vector.embeddings.retry_config import (
    EMBEDDING_ATTEMPT_TIMEOUT,
    EMBEDDING_RETRY_WINDOW,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["LiteLLMEmbeddingEngine", "OpenAICompatibleEmbeddingEngine"])
async def test_hung_attempt_is_cancelled_before_retry_succeeds(name, monkeypatch):
    module = importlib.import_module(f"cognee.infrastructure.databases.vector.embeddings.{name}")
    cls = getattr(module, name)
    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    monkeypatch.setattr(module, "EMBEDDING_ATTEMPT_TIMEOUT", 0.01)
    with patch.object(cls, "get_tokenizer", return_value=SimpleNamespace(count_tokens=lambda t: 1)):
        engine = cls(
            model="test-model", dimensions=2, endpoint="http://localhost:1/v1", api_key="test"
        )
    cancelled = False
    calls = 0

    async def request(**kwargs):
        nonlocal calls, cancelled
        calls += 1
        if calls == 1:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled = True
        assert cancelled
        item = (
            {"embedding": [1.0, 0.0]}
            if name.startswith("Lite")
            else SimpleNamespace(embedding=[1.0, 0.0])
        )
        return SimpleNamespace(data=[item], usage=SimpleNamespace(prompt_tokens=1))

    if name.startswith("Lite"):
        monkeypatch.setattr(module.litellm, "aembedding", request)
    else:
        monkeypatch.setattr(engine._client.embeddings, "create", request)
        assert engine._client.max_retries == 0
    engine.embed_text = engine.embed_text.retry_with(wait=wait_none()).__get__(engine)
    try:
        result = await engine.embed_text(["hello"])
        assert result == [[1.0, 0.0]]
        assert calls == 2
    finally:
        if not name.startswith("Lite"):
            await engine._client.close()


def test_timeout_leaves_retry_room_and_bounds_unresponsive_endpoint():
    assert 2 * EMBEDDING_ATTEMPT_TIMEOUT + 3 < EMBEDDING_RETRY_WINDOW
    assert EMBEDDING_RETRY_WINDOW + EMBEDDING_ATTEMPT_TIMEOUT < 300


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["LiteLLMEmbeddingEngine", "OpenAICompatibleEmbeddingEngine"])
async def test_retry_after_hung_http_response_opens_another_connection(name, monkeypatch):
    """Exercise the real HTTP clients; no provider or real API key is used."""
    module = importlib.import_module(f"cognee.infrastructure.databases.vector.embeddings.{name}")
    cls = getattr(module, name)
    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    monkeypatch.setattr(module, "EMBEDDING_ATTEMPT_TIMEOUT", 0.2)
    peers = []
    writers = []
    first_closed = asyncio.Event()

    async def serve(reader, writer):
        writers.append(writer)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in header.splitlines()
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(length)
            peers.append(writer.get_extra_info("peername"))
            if len(peers) == 1:
                await reader.read()
                first_closed.set()
                return
            body = json.dumps(
                {
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    with patch.object(cls, "get_tokenizer", return_value=SimpleNamespace(count_tokens=lambda t: 1)):
        engine = cls(
            model="text-embedding-3-small",
            dimensions=2,
            endpoint=f"http://127.0.0.1:{port}/v1",
            api_key="local-test",
        )
    engine.embed_text = engine.embed_text.retry_with(wait=wait_none()).__get__(engine)
    try:
        assert await asyncio.wait_for(engine.embed_text(["hello"]), timeout=5) == [[1.0, 0.0]]
        await asyncio.wait_for(first_closed.wait(), timeout=1)
        assert len(peers) == 2
        assert peers[0] != peers[1]
    finally:
        server.close()
        await server.wait_closed()
        for writer in writers:
            writer.close()
        if not name.startswith("Lite"):
            await engine._client.close()
