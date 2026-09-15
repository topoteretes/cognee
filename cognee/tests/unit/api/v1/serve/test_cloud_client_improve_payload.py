"""CloudClient.improve must put every improve() option on the wire.

improve()'s remote path forwards all options to the client; the drop the
orchestration tests couldn't see lived here, in the payload the client
actually POSTs. These tests pin that layer.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient


class _FakeResponse:
    status = 200

    async def json(self):
        return {"stages": []}

    async def text(self):
        return ""


class _FakePost:
    def __init__(self, recorder, url, json):
        recorder.append((url, json))

    async def __aenter__(self):
        return _FakeResponse()

    async def __aexit__(self, *exc):
        return False


def _client_with_recorder():
    client = CloudClient.__new__(CloudClient)
    client.service_url = "http://server"
    posts = []
    session = type(
        "FakeSession", (), {"post": lambda self, url, json: _FakePost(posts, url, json)}
    )()
    client._get_session = AsyncMock(return_value=session)
    return client, posts


@pytest.mark.asyncio
async def test_improve_forwards_every_option():
    client, posts = _client_with_recorder()

    await client.improve(
        "docs",
        node_name=["n1"],
        session_ids=["s1", "s2"],
        build_global_context_index=True,
        build_truth_subspace=True,
        feedback_alpha=0.4,
        run_in_background=True,
        extraction_tasks=["extract_a"],
        enrichment_tasks=["enrich_b"],
        data="seed text",
    )

    url, payload = posts[0]
    assert url == "http://server/api/v1/improve"
    assert payload == {
        "dataset_name": "docs",
        "node_name": ["n1"],
        "session_ids": ["s1", "s2"],
        "build_global_context_index": True,
        "build_truth_subspace": True,
        "feedback_alpha": 0.4,
        "run_in_background": True,
        "extraction_tasks": ["extract_a"],
        "enrichment_tasks": ["enrich_b"],
        "data": "seed text",
    }


@pytest.mark.asyncio
async def test_improve_refuses_options_that_cannot_cross_the_wire():
    """Task objects and the db-config overrides have no wire form; a loud
    error beats the server silently running defaults."""
    client, posts = _client_with_recorder()

    with pytest.raises(ValueError, match="registry task names"):
        await client.improve("docs", extraction_tasks=[lambda: None])
    with pytest.raises(ValueError, match="node_type"):
        await client.improve("docs", node_type=object())
    assert posts == []


def test_the_key_partition_covers_the_memify_passthrough_surface():
    """Every MEMIFY_PASSTHROUGH_KEYS entry is either forwarded or refused —
    a key added to the surface must never be silently dropped, and a typo in
    the serializable list must not invent a key off the surface."""
    from cognee.api.v1.serve.cloud_client import (
        _SERIALIZABLE_MEMIFY_TASK_KEYS,
        _UNSERIALIZABLE_MEMIFY_KEYS,
    )
    from cognee.modules.improve import MEMIFY_PASSTHROUGH_KEYS

    handled = {*_SERIALIZABLE_MEMIFY_TASK_KEYS, "data", *_UNSERIALIZABLE_MEMIFY_KEYS}
    assert handled == set(MEMIFY_PASSTHROUGH_KEYS)


@pytest.mark.asyncio
async def test_improve_defaults_send_only_the_dataset():
    client, posts = _client_with_recorder()
    dataset_id = uuid4()

    await client.improve(dataset_id)

    _, payload = posts[0]
    assert payload == {"dataset_id": str(dataset_id)}
