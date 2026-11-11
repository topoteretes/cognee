import asyncio
import base64

import numpy as np
import pytest

from cognee.infrastructure.databases.vector.embeddings.array_format import (
    arrays_requested,
    embed_float32,
)
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.EmbeddingGateway import EmbeddingGateway
from cognee.infrastructure.databases.vector.embeddings.encoding import decode_embedding


class Engine:
    model = "test"

    def __init__(self):
        self.calls = []
        self.native = []

    def get_batch_size(self):
        return 2

    async def embed_text(self, texts):
        await asyncio.sleep(0)
        self.calls.append(texts)
        vector = np.array([0.123456789, -0.987654321, 1e30], dtype="<f4")
        wire = base64.b64encode(vector.tobytes()).decode()
        result = [decode_embedding(wire, "base64") for _ in texts]
        self.native.extend(result)
        return EmbeddingResult(result, tokens=len(texts))


@pytest.mark.asyncio
async def test_arrays_survive_provider_gateway_batches_without_round_trip(monkeypatch):
    engine = Engine()
    gateway = EmbeddingGateway(engine)
    usage = []

    async def record(tokens):
        usage.append(tokens)

    monkeypatch.setattr(gateway, "_record_usage", record)
    output = await gateway.embed_text_float32([str(i) for i in range(5)])
    assert [len(call) for call in engine.calls] == [2, 2, 1]
    assert usage == [5]
    assert all(result is original for result, original in zip(output, engine.native))
    assert all(result.dtype == np.float32 for result in output)
    assert not arrays_requested.get()


@pytest.mark.asyncio
async def test_simultaneous_list_and_array_consumers_are_independent(monkeypatch):
    gateway = EmbeddingGateway(Engine())
    monkeypatch.setattr(gateway, "_record_usage", _ignore_usage)
    arrays, lists = await asyncio.gather(
        gateway.embed_text_float32(["a"]), gateway.embed_text(["b"])
    )
    assert isinstance(arrays[0], np.ndarray)
    assert isinstance(lists[0], list)
    np.testing.assert_array_equal(arrays[0], np.array(lists[0], dtype=np.float32))


async def _ignore_usage(tokens):
    pass


@pytest.mark.asyncio
async def test_json_vectors_round_once_after_provider_pooling():
    class Pooled:
        async def embed_text(self, texts):
            # Provider recursion pools its original float64 JSON values first.
            left = np.array([1e38, 0.1000000001], dtype=np.float64)
            right = np.array([1e38, 0.3000000001], dtype=np.float64)
            return EmbeddingResult([((left + right) / 2).tolist()], tokens=1)

    engine = Pooled()
    expected = np.asarray((await engine.embed_text(["x"]))[0], dtype=np.float32)
    (actual,) = await embed_float32(engine, ["x"])
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.asyncio
async def test_array_request_scope_is_reset_on_failure():
    class Broken:
        async def embed_text(self, texts):
            assert arrays_requested.get()
            raise ValueError("provider failed")

    with pytest.raises(ValueError, match="provider failed"):
        await embed_float32(Broken(), ["x"])
    assert not arrays_requested.get()
